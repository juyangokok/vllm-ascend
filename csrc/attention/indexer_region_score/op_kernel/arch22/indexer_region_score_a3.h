// Copyright (c) 2026 Huawei Technologies Co., Ltd.
// Licensed under the CANN Open Software License Agreement Version 2.0.
#ifndef INDEXER_REGION_SCORE_A3_H
#define INDEXER_REGION_SCORE_A3_H

#include "kernel_operator.h"
#include "a3_tiling_data.h"

namespace IndexerRegionScoreA3 {
using namespace AscendC;

// A row is owned by one Cube/AIV pair. Only a bounded H-by-128 dot tile
// crosses GM. Scores retain the source's quartet reduction order; no floating
// atomics or cross-core floating reductions are used.
class A3IndexerRegionScore {
public:
    __aicore__ inline void Init(GM_ADDR query, GM_ADDR weights, GM_ADDR key,
        GM_ADDR lengths, GM_ADDR table, GM_ADDR cu, GM_ADDR output, GM_ADDR ids,
        GM_ADDR userWorkspace, const PagedTilingData &data, TPipe *pipe)
    {
        data_ = data;
        pipe_ = pipe;
        query_.SetGlobalBuffer(reinterpret_cast<__gm__ bfloat16_t *>(query));
        key_.SetGlobalBuffer(reinterpret_cast<__gm__ bfloat16_t *>(key));
        weightsFloat_.SetGlobalBuffer(reinterpret_cast<__gm__ float *>(weights));
        weightsBits_.SetGlobalBuffer(reinterpret_cast<__gm__ uint16_t *>(weights));
        weightsHalf_.SetGlobalBuffer(reinterpret_cast<__gm__ half *>(weights));
        lengths_.SetGlobalBuffer(reinterpret_cast<__gm__ int32_t *>(lengths));
        table_.SetGlobalBuffer(reinterpret_cast<__gm__ int32_t *>(table));
        cu_.SetGlobalBuffer(reinterpret_cast<__gm__ int32_t *>(cu));
        ids_.SetGlobalBuffer(reinterpret_cast<__gm__ int32_t *>(ids));
        scores_.SetGlobalBuffer(reinterpret_cast<__gm__ float *>(output));
        if (data_.batchTokens != 0) {
            scores_.SetGlobalBuffer(reinterpret_cast<__gm__ float *>(userWorkspace + data_.scorePoolOffset));
        }
        core_ = GetBlockIdx();
        if ASCEND_IS_AIV {
            core_ /= 2;
            sub_ = GetSubBlockIdx();
        }
        bridge_.SetGlobalBuffer(reinterpret_cast<__gm__ float *>(
            userWorkspace + data_.reservedStride0 + core_ * kBridgeBytesPerCore));
        if ASCEND_IS_AIC {
            InitCube();
        } else {
            InitVector();
        }
    }

    __aicore__ inline void Process()
    {
        if ASCEND_IS_AIC {
            CubeRows();
        } else {
            VectorRows();
        }
        PipeBarrier<PIPE_ALL>();
        SyncAll<false>();
        if ASCEND_IS_AIV {
            SelectRows();
        }
        PipeBarrier<PIPE_ALL>();
    }

private:
    static constexpr uint16_t kReady = 0;
    static constexpr uint16_t kFree = 1;
    static constexpr uint32_t kSelectionWindow = 4096;

    __aicore__ inline uint32_t Min(uint32_t a, uint32_t b) const { return a < b ? a : b; }
    __aicore__ inline uint32_t Align16(uint32_t n) const { return (n + 15U) / 16U * 16U; }

    __aicore__ inline bool Request(uint32_t token, uint32_t &request, uint32_t &visible) const
    {
        visible = 0;
        for (uint32_t n = 0; n < data_.numRequests; ++n) {
            const int32_t begin = cu_.GetValue(n), end = cu_.GetValue(n + 1);
            if (begin < 0 || end < begin || static_cast<uint32_t>(end) > data_.totalTokens) continue;
            if (token >= static_cast<uint32_t>(begin) && token < static_cast<uint32_t>(end)) {
                request = n;
                const int64_t position = static_cast<int64_t>(lengths_.GetValue(n)) - (end - begin) + token - begin;
                if (position > 0) visible = Min(static_cast<uint32_t>(position / data_.regionSize), data_.seqK);
                return true;
            }
        }
        return false;
    }

    __aicore__ inline uint64_t ScoreBase(uint32_t group, uint32_t token) const
    {
        const uint32_t row = data_.batchTokens == 0 ? token : token - data_.batchBegin;
        return static_cast<uint64_t>(group) * data_.outputStrideGroup +
            static_cast<uint64_t>(row) * data_.outputStrideToken;
    }

    __aicore__ inline float Weight(uint32_t group, uint32_t token, uint32_t head) const
    {
        const uint64_t offset = static_cast<uint64_t>(token) * data_.weightStrideToken +
            static_cast<uint64_t>(group) * data_.weightStrideGroup + head;
        float value;
        if (data_.weightDtype == 0) {
            value = weightsFloat_.GetValue(offset);
        } else if (data_.weightDtype == 1) {
            union { uint32_t bits; float number; } converted;
            converted.bits = static_cast<uint32_t>(weightsBits_.GetValue(offset)) << 16U;
            value = converted.number;
        } else {
            value = static_cast<float>(weightsHalf_.GetValue(offset));
        }
        return data_.headScale == 1.0F ? value : value * data_.headScale;
    }

    __aicore__ inline void InitCube()
    {
        pipe_->InitBuffer(qL1Buffer_, kHeadCapacity * kDimension * sizeof(bfloat16_t));
        pipe_->InitBuffer(kL1Buffer_, kRegionTile * kDimension * sizeof(bfloat16_t));
        pipe_->InitBuffer(qL0Buffer_, kHeadCapacity * kDimension * sizeof(bfloat16_t));
        pipe_->InitBuffer(kL0Buffer_, kRegionTile * kDimension * sizeof(bfloat16_t));
        pipe_->InitBuffer(cL0Buffer_, kHeadCapacity * kRegionTile * sizeof(float));
        qL1_ = qL1Buffer_.Get<bfloat16_t>();
        kL1_ = kL1Buffer_.Get<bfloat16_t>();
        qL0_ = qL0Buffer_.Get<bfloat16_t>();
        kL0_ = kL0Buffer_.Get<bfloat16_t>();
        cL0_ = cL0Buffer_.Get<float>();
        SetFlag<HardEvent::M_MTE2>(0);
        SetFlag<HardEvent::FIX_M>(0);
    }

    __aicore__ inline void InitVector()
    {
        pipe_->InitBuffer(dotBuffer_, kHeadCapacity * kVectorColumns * sizeof(float));
        pipe_->InitBuffer(accBuffer_, kVectorColumns * sizeof(float));
        pipe_->InitBuffer(partBuffer_, kVectorColumns * sizeof(float));
        pipe_->InitBuffer(weightBuffer_, kVectorColumns * sizeof(float));
        pipe_->InitBuffer(windowBuffer_, kSelectionWindow * sizeof(float));
        pipe_->InitBuffer(heapScoreBuffer_, kMaxTopK * sizeof(float));
        pipe_->InitBuffer(heapIdBuffer_, kMaxTopK * sizeof(int32_t));
        dots_ = dotBuffer_.Get<float>();
        acc_ = accBuffer_.Get<float>();
        partial_ = partBuffer_.Get<float>();
        weight_ = weightBuffer_.Get<float>();
        window_ = windowBuffer_.Get<float>();
        heapScores_ = heapScoreBuffer_.Get<float>();
        heapIds_ = heapIdBuffer_.Get<int32_t>();
    }

    __aicore__ inline void LoadCubeTile(uint32_t group, uint32_t token,
        uint32_t request, uint32_t regionStart, uint32_t count)
    {
        WaitFlag<HardEvent::M_MTE2>(0);
        const uint32_t m = Align16(data_.headsPerGroup);
        InitConstValueParams<uint16_t> clearQ(1, static_cast<uint16_t>(m * kDimension / 16), 0, 0);
        InitConstValueParams<uint16_t> clearK(1, static_cast<uint16_t>(kRegionTile * kDimension / 16), 0, 0);
        InitConstValue(qL1_.ReinterpretCast<uint16_t>(), clearQ);
        InitConstValue(kL1_.ReinterpretCast<uint16_t>(), clearK);
        PipeBarrier<PIPE_MTE2>();
        Nd2NzParams qCopy{};
        qCopy.ndNum = 1;
        qCopy.nValue = data_.headsPerGroup;
        qCopy.dValue = kDimension;
        qCopy.srcDValue = kDimension;
        qCopy.dstNzC0Stride = m;
        qCopy.dstNzNStride = 1;
        DataCopy(qL1_, query_[static_cast<uint64_t>(token) * data_.qStrideToken +
            static_cast<uint64_t>(group) * data_.qStrideGroup], qCopy);
        // Key is supplied as [regions, D]; each small copy preserves its NZ row.
        Nd2NzParams keyCopy{};
        keyCopy.ndNum = 1;
        keyCopy.nValue = 1;
        keyCopy.dValue = kDimension;
        keyCopy.srcDValue = kDimension;
        keyCopy.dstNzC0Stride = kRegionTile;
        keyCopy.dstNzNStride = 1;
        for (uint32_t r = 0; r < count; ++r) {
            const uint32_t logical = regionStart + r;
            const int32_t page = table_.GetValue(static_cast<uint64_t>(request) * data_.blockTableStrideRequest +
                logical / data_.summariesPerPage);
            if (page < 0 || static_cast<uint32_t>(page) >= data_.numPhysicalPages) continue;
            const uint64_t offset = static_cast<uint64_t>(page) * data_.cacheStridePage +
                (logical % data_.summariesPerPage) * data_.cacheStrideRegion;
            DataCopy(kL1_[r * 16], key_[offset], keyCopy);
        }
        SetFlag<HardEvent::MTE2_MTE1>(0);
        WaitFlag<HardEvent::MTE2_MTE1>(0);
        LoadData2DParams loadQ{};
        loadQ.repeatTimes = kDimension / 16;
        loadQ.srcStride = m / 16;
        loadQ.ifTranspose = false;
        for (uint32_t block = 0; block < m / 16; ++block) {
            LoadData(qL0_[block * kDimension * 16], qL1_[block * 256], loadQ);
        }
        LoadData2DParams loadK{};
        loadK.repeatTimes = kRegionTile * kDimension / 256;
        loadK.srcStride = 1;
        loadK.ifTranspose = false;
        LoadData(kL0_, kL1_, loadK);
        SetFlag<HardEvent::MTE1_M>(0);
        WaitFlag<HardEvent::MTE1_M>(0);
    }

    __aicore__ inline void CubeRows()
    {
        const uint32_t rows = data_.groups * data_.batchCount;
        for (uint32_t row = core_; row < rows; row += data_.usedCubeCores) {
            const uint32_t group = row / data_.batchCount;
            const uint32_t token = data_.batchBegin + row % data_.batchCount;
            uint32_t request = 0, visible = 0;
            if (!Request(token, request, visible)) continue;
            for (uint32_t r = 0; r < visible; r += kRegionTile) {
                LoadCubeTile(group, token, request, r, Min(kRegionTile, visible - r));
                WaitFlag<HardEvent::FIX_M>(0);
                MmadParams mm{};
                mm.m = Align16(data_.headsPerGroup);
                mm.n = kRegionTile;
                mm.k = kDimension;
                mm.cmatrixInitVal = true;
                Mmad(cL0_, qL0_, kL0_, mm);
                SetFlag<HardEvent::M_MTE2>(0);
                SetFlag<HardEvent::M_FIX>(0);
                WaitFlag<HardEvent::M_FIX>(0);
                CrossCoreWaitFlag(kFree);
                FixpipeParamsV220 fix{};
                fix.mSize = Align16(data_.headsPerGroup);
                fix.nSize = kRegionTile;
                fix.srcStride = fix.mSize;
                fix.dstStride = kRegionTile;
                fix.quantPre = QuantMode_t::NoQuant;
                fix.reluEn = true;
                Fixpipe<float, float, CFG_ROW_MAJOR>(bridge_, cL0_, fix);
                CrossCoreSetFlag<2, PIPE_FIX>(kReady);
                SetFlag<HardEvent::FIX_M>(0);
            }
        }
        // Drain the initial/final free permit, including pairs with no rows.
        CrossCoreWaitFlag(kFree);
        WaitFlag<HardEvent::M_MTE2>(0);
        WaitFlag<HardEvent::FIX_M>(0);
    }

    __aicore__ inline void Reduce(uint32_t group, uint32_t token)
    {
        for (uint32_t h = 0; h < data_.headsPerGroup; h += 4) {
            auto partial = h == 0 ? acc_ : partial_;
            Muls(partial, dots_[h * kVectorColumns], Weight(group, token, h), kVectorColumns);
            PipeBarrier<PIPE_V>();
            for (uint32_t j = 1; j < 4; ++j) {
                Duplicate(weight_, Weight(group, token, h + j), kVectorColumns);
                PipeBarrier<PIPE_V>();
                MulAddDst(partial, dots_[(h + j) * kVectorColumns], weight_, kVectorColumns);
                PipeBarrier<PIPE_V>();
            }
            if (h != 0) {
                Add(acc_, acc_, partial_, kVectorColumns);
            }
            PipeBarrier<PIPE_V>();
        }
    }

    __aicore__ inline void VectorRows()
    {
        CrossCoreSetFlag<2, PIPE_MTE2>(kFree);
        const uint32_t rows = data_.groups * data_.batchCount;
        for (uint32_t row = core_; row < rows; row += data_.usedCubeCores) {
            const uint32_t group = row / data_.batchCount;
            const uint32_t token = data_.batchBegin + row % data_.batchCount;
            uint32_t request = 0, visible = 0;
            if (!Request(token, request, visible)) continue;
            for (uint32_t r = 0; r < visible; r += kRegionTile) {
                CrossCoreWaitFlag(kReady);
                DataCopyParams copy{};
                copy.blockCount = data_.headsPerGroup;
                copy.blockLen = kVectorColumns * sizeof(float) / 32;
                copy.srcStride = kVectorColumns * sizeof(float) / 32;
                DataCopy(dots_, bridge_[sub_ * kVectorColumns], copy);
                SetFlag<HardEvent::MTE2_V>(0);
                WaitFlag<HardEvent::MTE2_V>(0);
                // All readers have completed before Cube may overwrite GM.
                CrossCoreSetFlag<2, PIPE_MTE2>(kFree);
                Reduce(group, token);
                // dots_ is consumed by V and overwritten by the next MTE2
                // copy. A tail's second AIV can skip its output store, so
                // MTE3 completion cannot serve as this buffer's free event.
                SetFlag<HardEvent::V_MTE2>(0);
                WaitFlag<HardEvent::V_MTE2>(0);
                const uint32_t start = r + sub_ * kVectorColumns;
                if (start < visible) {
                    const uint32_t count = Min(kVectorColumns, visible - start);
                    SetFlag<HardEvent::V_MTE3>(0);
                    WaitFlag<HardEvent::V_MTE3>(0);
                    DataCopyExtParams store{1, count * 4U, 0, 0, 0};
                    DataCopyPad(scores_[ScoreBase(group, token) + start], acc_, store);
                    SetFlag<HardEvent::MTE3_V>(0);
                    WaitFlag<HardEvent::MTE3_V>(0);
                }
            }
        }
    }

    // Better means a higher score or the smaller logical ID at exact ties.
    __aicore__ inline bool Better(float a, int32_t ai, float b, int32_t bi) const
    {
        return a > b || (a == b && ai < bi);
    }

    __aicore__ inline void Swap(uint32_t a, uint32_t b)
    {
        const float score = heapScores_.GetValue(a);
        const int32_t id = heapIds_.GetValue(a);
        heapScores_.SetValue(a, heapScores_.GetValue(b));
        heapIds_.SetValue(a, heapIds_.GetValue(b));
        heapScores_.SetValue(b, score);
        heapIds_.SetValue(b, id);
    }

    __aicore__ inline void Push(float score, int32_t id, uint32_t &size)
    {
        uint32_t at = size++;
        heapScores_.SetValue(at, score);
        heapIds_.SetValue(at, id);
        while (at != 0) {
            const uint32_t parent = (at - 1) / 2;
            if (!Better(heapScores_.GetValue(parent), heapIds_.GetValue(parent), score, id)) break;
            Swap(parent, at);
            at = parent;
        }
    }

    __aicore__ inline void ReplaceRoot(float score, int32_t id, uint32_t size)
    {
        heapScores_.SetValue(0, score);
        heapIds_.SetValue(0, id);
        uint32_t at = 0;
        while (at * 2 + 1 < size) {
            uint32_t child = at * 2 + 1;
            if (child + 1 < size && Better(heapScores_.GetValue(child), heapIds_.GetValue(child),
                heapScores_.GetValue(child + 1), heapIds_.GetValue(child + 1))) ++child;
            if (!Better(heapScores_.GetValue(at), heapIds_.GetValue(at),
                heapScores_.GetValue(child), heapIds_.GetValue(child))) break;
            Swap(at, child);
            at = child;
        }
    }

    __aicore__ inline void SiftDownIds(uint32_t at, uint32_t size)
    {
        const int32_t id = heapIds_.GetValue(at);
        while (at * 2 + 1 < size) {
            uint32_t child = at * 2 + 1;
            if (child + 1 < size && heapIds_.GetValue(child + 1) > heapIds_.GetValue(child)) ++child;
            const int32_t childId = heapIds_.GetValue(child);
            if (id >= childId) break;
            heapIds_.SetValue(at, childId);
            at = child;
        }
        heapIds_.SetValue(at, id);
    }

    __aicore__ inline void SortIds(uint32_t size)
    {
        // Sort only integer IDs. A max heap yields ascending output with
        // bounded O(K log K) work, including large equal-score tie sets.
        for (uint32_t root = size / 2; root != 0; --root) SiftDownIds(root - 1, size);
        for (uint32_t end = size; end > 1; --end) {
            const int32_t largest = heapIds_.GetValue(0);
            heapIds_.SetValue(0, heapIds_.GetValue(end - 1));
            heapIds_.SetValue(end - 1, largest);
            SiftDownIds(0, end - 1);
        }
    }

    __aicore__ inline void SelectRows()
    {
        const uint32_t k = data_.selectionConfig & 0xFFFFU;
        const uint32_t rows = data_.groups * data_.batchCount;
        const uint32_t vectorCore = GetBlockIdx();
        for (uint32_t row = vectorCore; row < rows; row += data_.usedCubeCores * 2) {
            const uint32_t group = row / data_.batchCount;
            const uint32_t token = data_.batchBegin + row % data_.batchCount;
            uint32_t request = 0, visible = 0;
            (void)Request(token, request, visible);
            uint32_t size = 0;
            if (visible <= k) {
                for (uint32_t i = 0; i < visible; ++i) heapIds_.SetValue(i, i);
                size = visible;
            } else {
                const uint64_t base = ScoreBase(group, token);
                for (uint32_t begin = 0; begin < visible; begin += kSelectionWindow) {
                    const uint32_t count = Min(kSelectionWindow, visible - begin);
                    DataCopyExtParams copy{1, count * 4U, 0, 0, 0};
                    DataCopyPadExtParams<float> pad{false, 0, 0, 0};
                    DataCopyPad(window_, scores_[base + begin], copy, pad);
                    SetFlag<HardEvent::MTE2_S>(0);
                    WaitFlag<HardEvent::MTE2_S>(0);
                    for (uint32_t i = 0; i < count; ++i) {
                        const float score = window_.GetValue(i);
                        const int32_t id = static_cast<int32_t>(begin + i);
                        if (size < k) {
                            Push(score, id, size);
                        } else if (Better(score, id, heapScores_.GetValue(0), heapIds_.GetValue(0))) {
                            ReplaceRoot(score, id, size);
                        }
                    }
                }
                SortIds(size);
            }
            for (uint32_t i = size; i < k; ++i) heapIds_.SetValue(i, -1);
            SetFlag<HardEvent::S_MTE3>(0);
            WaitFlag<HardEvent::S_MTE3>(0);
            const uint64_t out = (static_cast<uint64_t>(group) * data_.totalTokens + token) * k;
            DataCopyExtParams copy{1, k * 4U, 0, 0, 0};
            DataCopyPad(ids_[out], heapIds_, copy);
            SetFlag<HardEvent::MTE3_S>(0);
            WaitFlag<HardEvent::MTE3_S>(0);
        }
    }

    PagedTilingData data_{};
    TPipe *pipe_ = nullptr;
    uint32_t core_ = 0, sub_ = 0;
    GlobalTensor<bfloat16_t> query_, key_;
    GlobalTensor<float> weightsFloat_, scores_, bridge_;
    GlobalTensor<uint16_t> weightsBits_;
    GlobalTensor<half> weightsHalf_;
    GlobalTensor<int32_t> lengths_, table_, cu_, ids_;
    TBuf<TPosition::A1> qL1Buffer_;
    TBuf<TPosition::B1> kL1Buffer_;
    TBuf<TPosition::A2> qL0Buffer_;
    TBuf<TPosition::B2> kL0Buffer_;
    TBuf<TPosition::CO1> cL0Buffer_;
    TBuf<TPosition::VECCALC> dotBuffer_, accBuffer_, partBuffer_, weightBuffer_;
    TBuf<TPosition::VECCALC> windowBuffer_, heapScoreBuffer_, heapIdBuffer_;
    LocalTensor<bfloat16_t> qL1_, kL1_, qL0_, kL0_;
    LocalTensor<float> cL0_, dots_, acc_, partial_, weight_, window_, heapScores_;
    LocalTensor<int32_t> heapIds_;
};
} // namespace IndexerRegionScoreA3
#endif
