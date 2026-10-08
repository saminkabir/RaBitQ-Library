#include <atomic>
#include <random>
#include <stdlib.h>
#include <assert.h>
#include <unordered_set>
#include <list>
#include <memory>
#include "io.h"
#include <type_traits>
#include <unordered_map>
#include <map>
#include <algorithm>
#include <fstream>
#include <array>
#include <mutex>
#include <omp.h>
#include <queue>
#include "heap.h"
#include "distance.h"
#include "oqg.h"

#define MULTI_START

namespace gg {

// For analyses only, enable it will make search much slower
///constexpr static bool timing = false;

template<size_t batchSize, size_t numCentroids, size_t numSubspaces, size_t dim, size_t numLevel0Edges>
class MultiNodesOQG {
public:
    using id_t = unsigned int;
    using dist_t = float;
    using code_t = uint8_t;
    using qdist_t = uint8_t;
    using qdist16_t = uint16_t;
    constexpr static size_t level0NodeBatchSize = batchSize / numLevel0Edges;

    constexpr static bool enableNumNeighborsSerialization = false;
    size_t adjustedSpaceNum = 0;

    template<size_t numEdges>
    struct Edge {
        id_t neighbors[numEdges];
        code_t neighborsCode[numSubspaces*numEdges];
        void load(std::ifstream &ifs) {
            readArray(ifs, neighbors, numEdges);
            readArray(ifs, neighborsCode, numSubspaces*numEdges);
        }
        void save(std::ofstream &ofs) {
            writeArray(ofs, neighbors, numEdges);
            writeArray(ofs, neighborsCode, numSubspaces*numEdges);
        }
    };

    struct Node {
        std::vector<Edge<batchSize>> edges;
        Edge<numLevel0Edges> level0Edges;
        Node(size_t levels){
            edges.resize(levels);
            edges.shrink_to_fit();
        }
        Node() = default;
        void load(std::ifstream &ifs) {
            size_t levels;
            readValue(ifs, levels);
            edges.resize(levels);
            level0Edges.load(ifs);
            for(int curLevel = 1; curLevel <= levels; ++curLevel) {
                edges[curLevel - 1].load(ifs);
            }
        }
        void save(std::ofstream &ofs) {
            size_t levels = getLevels();
            writeValue(ofs, levels);
            level0Edges.save(ofs);
            for(int curLevel = 1; curLevel <= levels; ++curLevel) {
                edges[curLevel - 1].save(ofs);
            }
        }
        size_t getLevels() const {
            return edges.size();
        }
    };



    
    struct IDGenerator {
        int maxID;
        std::vector<id_t> ids; 
        size_t cursor = 0; 
        std::mt19937 rng;  

        IDGenerator(int maxID_) : maxID(maxID_), ids(maxID_), rng(std::random_device{}()) {
            ids.shrink_to_fit();
            for (int i = 0; i < maxID; i++) {
                ids[i] = i;
            }
            reshuffle();
        }

        void reshuffle() {
            std::shuffle(ids.begin(), ids.end(), rng);
            cursor = 0;
        }

        std::vector<id_t> getIDs(int requiredNum, const std::vector<id_t>& avoids) {
            std::unordered_set<id_t> avoidSet(avoids.begin(), avoids.end());
            std::vector<id_t> result;
            result.reserve(requiredNum);

            while ((int)result.size() < requiredNum) {
                if (cursor >= ids.size()) {
                    reshuffle();
                }

                id_t candidate = ids[cursor++];
                if (avoidSet.find(candidate) == avoidSet.end()) {
                    result.push_back(candidate);
                }
            }

            return result;
        }

        std::vector<id_t> getIDsFromCandidates(
            int requiredNum,
            const std::vector<id_t>& avoids,
            const std::vector<id_t>& candidates
        ) {
            std::unordered_set<id_t> avoidSet(avoids.begin(), avoids.end());

            std::vector<id_t> filtered;
            filtered.reserve(candidates.size());
            for (auto c : candidates) {
                if (avoidSet.find(c) == avoidSet.end()) {
                    filtered.push_back(c);
                }
            }

            if ((int)filtered.size() <= requiredNum) {
                return filtered;
            }

            std::vector<id_t> result;
            result.reserve(requiredNum);

            std::random_device rd;
            std::mt19937 gen(rd());

            std::sample(filtered.begin(), filtered.end(),
                        std::back_inserter(result),
                        requiredNum,
                        gen);

            return result;
        }

    };

    template<typename MarkDType=unsigned short int>
    struct VisitedArray {
        std::vector<MarkDType> mark;
        MarkDType currentGen;       
        int N;               
        VisitedArray() = default;

        void init(int n){
            mark = std::vector<MarkDType>(n, 0);
            currentGen = 1;
            N = n;
        }

        void nextGeneration() {
            ++currentGen;
            if (currentGen == std::numeric_limits<MarkDType>::max()) {
                std::fill(mark.begin(), mark.end(), 0);
                currentGen = 1;
            }
        }

        void markVisited(int id) {
            mark[id] = currentGen;
        }

        bool isVisited(int id) {
            return mark[id] == currentGen;
        }
    };

    struct BorrowedPtr {
        const dist_t* p = nullptr;
        const dist_t* get() const { return p; }
        void reset(const dist_t* x) { p = x; }

        bool operator==(const BorrowedPtr& other) const {
            return p == other.p;
        }
        bool operator!=(const BorrowedPtr& other) const {
            return p != other.p;
        }
        bool operator==(std::nullptr_t) const {
            return p == nullptr;
        }
        bool operator!=(std::nullptr_t) const {
            return p != nullptr;
        }
    };

    // -------------------------------Variables Begin------------------------------
    std::vector<Node> nodes;
    BorrowedPtr rawVectors;; // optional serialize
    std::unique_ptr<IDGenerator> idGenerator; // not serialize
    static inline thread_local VisitedArray<unsigned short int> visitedArray; // not serialize
    std::vector<std::vector<size_t>> numNeighbors; // not serialize
    std::vector<id_t> toExternalID;
    alignas(64) dist_t pqCentroids[numCentroids*dim];
    alignas(64) dist_t lut[numCentroids*numSubspaces]; // not serialize
    alignas(64) qdist_t qLUT[numCentroids*numSubspaces]; // not serialize
    alignas(64) __m512i q8LUTr[4 * numSubspaces];
    alignas(64) qdist_t q16LUT[numCentroids*numSubspaces];
    alignas(64) __m512i q16LUTr[4 * numSubspaces];
    DISTFUNC<dist_t> distFunc; // not serialize
    size_t traversal = 0; // not serialize
    std::vector<std::mutex> nodesLock; // not serialize
    std::mutex globalLock; // not serialize

    size_t curElements, efConstruction, M, M0, MFar, maxLevel, maxLevelElementsCt = 0;
    id_t entryPoint;
    // -------------------------------Variables End------------------------------

    void transposePQCentroids(
        dist_t* pqCentroids,  // length = M * K * dsub, will be overwritten
        size_t M,             // numSubspaces
        size_t K,             // numCentroids
        size_t dsub           // subspaceDim
    ){
        std::vector<dist_t> buf(M * K * dsub);

        // input:  [K][M][dsub]  -> idx_in  = i*dim + s*dsub + k
        // output: [M][K][dsub]  -> idx_out = (s*K + i)*dsub + k
        for (size_t i = 0; i < K; ++i) {
            for (size_t s = 0; s < M; ++s) {
                const dist_t* src = pqCentroids + i * dim + s * dsub;
                dist_t* dst = buf.data() + (s * K + i) * dsub;
                std::memcpy(dst, src, dsub * sizeof(dist_t));
            }
        }

        std::memcpy(pqCentroids, buf.data(), buf.size() * sizeof(dist_t));
    }

    void load(std::ifstream &ifs) {
        L2Space *l2space = new L2Space(dim);
        distFunc = l2space->get_dist_func();
        delete l2space;

        readValue(ifs, curElements);
        readValue(ifs, efConstruction);
        readValue(ifs, M);
        readValue(ifs, M0);
        readValue(ifs, MFar);
        readValue(ifs, entryPoint);
        readValue(ifs, maxLevel);
        readValue(ifs, maxLevelElementsCt);

        nodes.resize(curElements);
        for(int curID = 0; curID < curElements; ++curID) {
            nodes[curID].load(ifs);
        }
        
        // Paralell: move to omp parallel 
        visitedArray.init(curElements);
        readVector(ifs, toExternalID);
        toExternalID.shrink_to_fit();
        readArray(ifs, pqCentroids, numCentroids*dim);
        transposePQCentroids(pqCentroids, numSubspaces, numCentroids, dim / numSubspaces);


        if constexpr (enableNumNeighborsSerialization) {
            size_t maxLevels;
            readValue(ifs, maxLevels);
            this->numNeighbors.resize(maxLevels + 1);
            for(size_t i = 0; i <= maxLevels; i++) {
                readVector(ifs, numNeighbors[i]);
            }
        }

        puts("-------------Graph Loaded-----------------");
        printf("Dim: %d, #objects: %d, efC: %d, R: %d, entryPoint: %d, maxLevel: %d, maxLevelEleCt: %d\n", 
            int(dim), int(curElements), int(efConstruction), int(M), int(entryPoint), int(maxLevel), int(maxLevelElementsCt));
        printf("PQ. #Centroids: %d, #Subspaces: %d\n", int(numCentroids), int(numSubspaces));
        puts("-------------Graph Loaded-----------------");
    }

    void setRawVectors(const dist_t *rawVectors) {
        // if(rawVectors != nullptr) {
        //     this->rawVectors = std::make_unique<dist_t[]>(curElements*dim);
        //     std::copy_n(rawVectors, curElements*dim, this->rawVectors.get());
        // }
        this->rawVectors.reset(rawVectors);
    }

    void save(std::ofstream &ofs) {
        writeValue(ofs, curElements);
        writeValue(ofs, efConstruction);
        writeValue(ofs, M);
        writeValue(ofs, M0);
        writeValue(ofs, MFar);
        writeValue(ofs, entryPoint);
        writeValue(ofs, maxLevel);
        writeValue(ofs, maxLevelElementsCt);

        printf("Saving %d elements\n", int(curElements));
        for (int curID = 0; curID < curElements; ++curID) {
            nodes[curID].save(ofs);
        }
        puts("Node Saved");

        writeVector(ofs, toExternalID);
        writeArray(ofs, pqCentroids, numCentroids*dim);

        if constexpr (enableNumNeighborsSerialization) {
            size_t maxLevels = numNeighbors.size() - 1;
            writeValue(ofs, maxLevels);
            for(size_t i = 0; i <= maxLevels; i++) {
                writeVector(ofs, numNeighbors[i]);
            }
        }
    }


    template <class DistT=dist_t>
    struct ID {
        id_t vecID;
        DistT dist;

        ID(id_t vecID, DistT d) : vecID(vecID), dist(d) {}
        ID() = default;

        bool operator<(const ID& rhs) const { return dist < rhs.dist; }
        bool operator>(const ID& rhs) const { return dist > rhs.dist; }
    };

    template <class DistT=dist_t>
    using MaxHeap = std::priority_queue<ID<DistT>, std::vector<ID<DistT>>, std::less<ID<DistT>>>;
    template <class DistT=dist_t>
    using MinHeap = std::priority_queue<ID<DistT>, std::vector<ID<DistT>>, std::greater<ID<DistT>>>;

    MultiNodesOQG(size_t efConstruction, size_t M) {
        curElements = 0;
        entryPoint = 0;
        maxLevel = 0;
        this->efConstruction = efConstruction;
        this->M = M;
        this->M0 = M;
        L2Space *l2space = new L2Space(dim);
        distFunc = l2space->get_dist_func();
        delete l2space;
    }

    MultiNodesOQG() = default;


    std::mt19937_64 level_generator_ = std::mt19937_64(std::random_device{}());
    std::mutex      random_mtx_;  
    size_t getRandomLevel() {
        static constexpr double levelDivisor = 64.0;
        static constexpr double mult = 1.0 / std::log(levelDivisor);

        std::uniform_real_distribution<double> dist(0.0, 1.0);
        double u;
        {
            std::lock_guard<std::mutex> lock(random_mtx_);
            u = dist(level_generator_);
        }

        if (u <= 0.0) {
            u = std::numeric_limits<double>::min();
        } else if (u >= 1.0) {
            u = std::nextafter(1.0, 0.0);
        }

        double r = -std::log(u) * mult;
        size_t level = static_cast<size_t>(r);

        return level;
    }


    template<typename DType>
    void transpose(const size_t row, const size_t col, DType *data) const {
        DType *dst = new DType[row*col];
        for (size_t i = 0; i < row; ++i) {
            for (size_t j = 0; j < col; ++j) {
                dst[j * row + i] = data[i * col + j];
            }
        }
        std::copy(dst, dst+row*col, data);
        delete[] dst;
    }

    

    template<size_t subspaceDim>
    dist_t computeDistance(
        const dist_t*  __restrict__ querySubspace, 
        const dist_t*  __restrict__ centroidSubspace
    ) const{
        dist_t distance = 0.0f;
        #pragma unroll
        for (size_t i = 0; i < subspaceDim; ++i) {
            dist_t diff = querySubspace[i] - centroidSubspace[i];
            distance += diff * diff;
        }
        return distance;
    }

    dist_t computePQDistance(
        const dist_t *lut,
        const code_t *codes,
        dist_t *dists
    ) {
        for(int idx = 0; idx < batchSize; ++idx) {
            dists[idx] = 0;
            auto curLUT = lut;
            for(int j = 0; j < numSubspaces; ++j) {
                dists[idx] += curLUT[(*codes)];
                codes++;
                curLUT += numCentroids;
            }
        }
    }


    void quantizeLUT() {
        dist_t qmin, qmax;
        const size_t K = numCentroids;
        const size_t M = numSubspaces;

        // Step 1: Trim trailing all-zero subspaces (check last subspace, then move left)
        if (adjustedSpaceNum == 0) {
            adjustedSpaceNum = M;
            while (adjustedSpaceNum >= 1) {
                const size_t s = adjustedSpaceNum - 1; // last active subspace
                bool isZero = true;
                for (size_t i = 0; i < K; ++i) {
                    if (std::abs(lut[s * K + i]) > 1e-20) {
                        isZero = false;
                        break;
                    }
                }
                if (isZero) adjustedSpaceNum--;
                else break;
            }
        }

        // Step 2: qmin = minimum over active subspaces
        qmin = std::numeric_limits<dist_t>::infinity();
        for (size_t s = 0; s < adjustedSpaceNum; ++s) {
            const dist_t* row = lut + s * K;
            for (size_t i = 0; i < K; ++i) {
                qmin = std::min(qmin, row[i]);
            }
        }

        // Step 3: qmax (your original definition)
        qmax = 0.0f;
        for (size_t s = 0; s < adjustedSpaceNum; ++s) {
            const dist_t* row = lut + s * K;

            dist_t columnMin  = std::numeric_limits<dist_t>::infinity();
            dist_t columnMean = 0.0f;

            for (size_t i = 0; i < K; ++i) {
                const dist_t v = row[i];
                columnMin  = std::min(columnMin, v);
                columnMean += (v / static_cast<dist_t>(K));
            }

            dist_t threshold = columnMin * static_cast<dist_t>(adjustedSpaceNum);
            qmax += std::min(columnMean, threshold);
        }

        constexpr qdist_t dtype_min = std::numeric_limits<qdist_t>::min();
        constexpr qdist_t dtype_max = std::numeric_limits<qdist_t>::max();

        debug(qmin != qmax && "if qmin == qmax, then do not need quantization");

        // Step 4: Quantize (subspace-major traversal)
        const dist_t invRange = dist_t(1.0) / (qmax - qmin);
        const dist_t scale = static_cast<dist_t>(dtype_max - dtype_min);

        auto* quant = qLUT;
        auto* original = lut;
        int lutSize = numCentroids*numSubspaces;
        while(lutSize--) {
            if(*original == 0) {
                *quant = 0;
            } else {
                dist_t ratio = ((*original) - qmin) / (qmax - qmin);
                ratio =  ratio < 1.0f ? ratio : 1.0f;
                *quant = static_cast<qdist_t>(std::round(ratio * (dtype_max-dtype_min)));
                *quant += dtype_min;
            }
            original++;
            quant++;
        }
    }



    template<size_t n>
    inline void minmax_finite_avx512_float(
        const float* lut,
        float& qmin, float& qmax
    ) {
        float minv = qmin;
        float maxv = qmax;

        __m512 vmin = _mm512_set1_ps(minv);
        __m512 vmax = _mm512_set1_ps(maxv);

        const __m512i expMask = _mm512_set1_epi32(0x7F800000);
        const __m512i absMask = _mm512_set1_epi32(0x7FFFFFFF);
        const __m512i expAll1 = _mm512_set1_epi32(0x7F800000);

        size_t i = 0;
        for (; i + 16 <= n; i += 16) {
            __m512 x = _mm512_loadu_ps(lut + i);

            __m512i xi = _mm512_castps_si512(x);
            __m512i xabs = _mm512_and_si512(xi, absMask);
            __m512i xexp = _mm512_and_si512(xabs, expMask);
            __mmask16 finite = _mm512_cmpneq_epi32_mask(xexp, expAll1);

            vmin = _mm512_mask_min_ps(vmin, finite, vmin, x);
            vmax = _mm512_mask_max_ps(vmax, finite, vmax, x);
        }

        alignas(64) float bufMin[16];
        alignas(64) float bufMax[16];
        _mm512_store_ps(bufMin, vmin);
        _mm512_store_ps(bufMax, vmax);
        for (int k = 0; k < 16; ++k) {
            minv = std::min(minv, bufMin[k]);
            maxv = std::max(maxv, bufMax[k]);
        }

        for (; i < n; ++i) {
            float v = lut[i];
            if (!std::isfinite(v)) continue;
            minv = std::min(minv, v);
            maxv = std::max(maxv, v);
        }

        qmin = minv;
        qmax = maxv;
    }


    void quantize16LUT() {
        using std::abs;
        using std::isfinite;

        dist_t qmin = std::numeric_limits<dist_t>::infinity();
        dist_t qmax = -std::numeric_limits<dist_t>::infinity();

        constexpr size_t numRows = numCentroids;
        constexpr size_t numCols = numSubspaces;

        minmax_finite_avx512_float<numRows * numCols>(lut, qmin, qmax);

        if ((!(qmax > qmin) || (qmax > qmin * 65536))) {
            quantize16LUT_saturated();
            // quantize16LUT_mean();
            // std::fill(q16LUT, q16LUT + numRows * numCols, static_cast<qdist_t>(128));
            return;
        }

        dist_t a = std::max(std::abs(qmin), std::abs(qmax));
        if (!std::isfinite((double)a) || a <= dist_t(0)) a = dist_t(1);

        constexpr qdist_t Qctr = 128; 
        constexpr int      Qrad = 127; 

        auto* quant    = q16LUT;
        auto* original = lut;
        int   lutSize  = int(numRows * numCols);

        while (lutSize--) {
            dist_t x = *original++;

            if (x == dist_t(0)) {
                *quant++ = Qctr;
                continue;
            }
            if (x >  a) x =  a;
            if (x < -a) x = -a;

            double r  = double(x) / double(a);       
            double qf = std::round(r * double(Qrad));   
            int  qi = (int )qf + (int)Qctr;

            if (qi < 0)   qi = 0;
            if (qi > 255) qi = 255;

            *quant++ = static_cast<qdist_t>(qi);
        }
    }


    void quantize16LUT_mean() {
        using std::abs;
        using std::isfinite;

        dist_t qmin = std::numeric_limits<dist_t>::infinity();
        dist_t qmax = 0;

        constexpr size_t numRows = numCentroids;
        constexpr size_t numCols = numSubspaces;

        if(adjustedSpaceNum == 0){
            adjustedSpaceNum = numCols;
            while (adjustedSpaceNum >= 1) {
                bool isZero = true;
                for (int i = 0; i < numRows; ++i) {
                    // lut is subspace-major: entry (subspace j, centroid i) at lut[j*numRows + i]
                    if (std::abs(lut[(adjustedSpaceNum - 1) * numRows + i]) > 0) {
                        isZero = false;
                        break;
                    }
                }
                if (isZero) {
                    adjustedSpaceNum--;
                } else {
                    break;
                }
            }
        }


        for (size_t j = 0; j < adjustedSpaceNum; ++j) {
            dist_t columnMin  = std::numeric_limits<dist_t>::infinity();
            dist_t columnMean = 0.0;

            for (size_t i = 0; i < numRows; ++i) {
                dist_t v = lut[j * numRows + i];
                if (!isfinite((double)v)) continue;
                columnMin  = std::min(columnMin, v);
                columnMean += v / dist_t(numRows);
            }

            dist_t threshold = columnMin * dist_t(adjustedSpaceNum);

            qmax += std::min(columnMean, threshold);

            qmin = std::min(qmin, columnMin);
   
        }
        if (!(qmax > qmin)) {
            std::fill(q16LUT, q16LUT + numRows * numCols, static_cast<qdist_t>(128));
            return;
        }

        dist_t a = std::max(abs(qmin), abs(qmax));
        if (a <= dist_t(0)) a = dist_t(1);

        constexpr qdist_t Qctr = 128;
        constexpr int      Qrad = 127;

        auto* quant    = q16LUT;
        auto* original = lut;
        int   lutSize  = int(numRows * numCols);

        while (lutSize--) {
            dist_t x = *original++;

            if (!isfinite((double)x)) {
                *quant++ = Qctr;
                continue;
            }

            if (x >  a) x =  a;
            if (x < -a) x = -a;

            if (x == dist_t(0)) {
                *quant++ = Qctr;
                continue;
            }

            double r  = double(x) / double(a);          // [-1,1]
            double qf = std::round(r * double(Qrad));   // [-127,127]
            long long qi = (long long)qf + (long long)Qctr;

            if (qi < 0)   qi = 0;
            if (qi > 255) qi = 255;


            *quant++ = static_cast<qdist_t>(qi);
        }

    }

    void quantize16LUT_saturated() {
        using std::isfinite;

        constexpr size_t numRows = numCentroids;
        constexpr size_t numCols = numSubspaces;

        constexpr double kPercentile = 0.9;

        if(adjustedSpaceNum == 0){
            adjustedSpaceNum = numCols;
            while (adjustedSpaceNum >= 1) {
                bool isZero = true;
                for (size_t i = 0; i < numRows; ++i) {
                    // lut is subspace-major: entry (subspace j, centroid i) at lut[j*numRows + i]
                    if (std::abs(lut[(adjustedSpaceNum - 1) * numRows + i]) > 1e-20) {
                        isZero = false;
                        break;
                    }
                }
                if (isZero) adjustedSpaceNum--;
                else break;
            }
        }

        dist_t qmin = std::numeric_limits<dist_t>::infinity();
        std::vector<dist_t> vals;
        vals.reserve(numRows * static_cast<size_t>(adjustedSpaceNum));

        for (size_t j = 0; j < adjustedSpaceNum; ++j) {
            const dist_t* col = lut + j * numRows;   // subspace-major
            for (size_t i = 0; i < numRows; ++i) {
                dist_t v = col[i];
                if (!isfinite((double)v)) continue;
                qmin = std::min(qmin, v);
                vals.push_back(v);
            }
        }

        if (vals.empty()) {
            quantize16LUT_mean();
            return;
            // std::fill(q16LUT, q16LUT + numRows * numCols, static_cast<qdist_t>(128));
            // return;
        }

        const size_t n = vals.size();
        const size_t kth = std::min(n - 1, static_cast<size_t>(std::floor(kPercentile * (n - 1))));
        std::nth_element(vals.begin(), vals.begin() + kth, vals.end());
        dist_t qmax = vals[kth];

        if (!(qmax > qmin) || (qmax > qmin * 65525)) {
            quantize16LUT_mean();
            return;
        }

        const double denom = double(qmax - qmin);
        auto* quant    = q16LUT;
        auto* original = lut;
        int   lutSize  = int(numRows * numCols);

        const double scale = 255.0 / double(denom);
        const double bias  = -double(qmin) * scale;

        while (lutSize--) {
            dist_t x = *original++;

            if (!std::isfinite(x)) {
                *quant++ = static_cast<qdist_t>(128); // invalid → center code
                continue;
            }

            // clamp to [qmin, qmax]
            if (x < qmin) x = qmin;
            if (x > qmax) x = qmax;

            // linear map → [0,255]
            double t = double(x) * scale + bias;    // already scaled
            int qi = std::lround(t);                // round to nearest int

            // saturate to [0,255]
            if (qi < 0)   qi = 0;
            if (qi > 255) qi = 255;

            *quant++ = static_cast<qdist_t>(qi);
        }
    }

    
    double lutLatency = 0;
    double lutConsLatency = 0;

    void computeLUT(const dist_t* query, bool is16 = false) {
        using clock = std::chrono::steady_clock;
        std::chrono::time_point<clock> t0;

        if constexpr(timing) {
            t0 = clock::now();
        }

        // The size of each subspace (dim / m)
        constexpr size_t subspaceDim = dim / numSubspaces;
        // for (size_t s = 0; s < numSubspaces; ++s) {
        //     const dist_t* qSub = query + s * subspaceDim;
        //     dist_t* out = lut + s * numCentroids;  

        //     for (size_t i = 0; i < numCentroids; ++i) {
        //         const dist_t* cSub = pqCentroids + i * dim + s * subspaceDim; 
        //         out[i] = computeDistance<subspaceDim>(qSub, cSub);
        //     }
        // }

        for (size_t s = 0; s < numSubspaces; ++s) {
            const dist_t* qSub = query + s*subspaceDim;
            dist_t* out = lut + s*numCentroids;

            const dist_t* centBase = pqCentroids + (s*numCentroids)*subspaceDim;   // pqCentroids is (M,K,dsub)

            for (size_t i = 0; i < numCentroids; ++i) {
                out[i] = computeDistance<subspaceDim>(qSub, centBase + i*subspaceDim);
            }
        }

        if constexpr(timing) {
            auto t1 = clock::now();
            lutConsLatency += std::chrono::duration_cast<std::chrono::nanoseconds>(t1 - t0).count();
        }


        if(!is16){
            quantizeLUT();
            //transpose(numCentroids, numSubspaces, qLUT);
            for (int col = 0; col < numSubspaces; ++col) {
                for (int j = 0; j < 4; ++j) {
                    q8LUTr[col * 4 + j] = _mm512_loadu_si512(reinterpret_cast<const void*>(
                        qLUT + col * 256 + j * 64));
                }
            }
        } else {
            quantize16LUT();
            //transpose(numCentroids, numSubspaces, q16LUT);
            for (int col = 0; col < numSubspaces; ++col) {
                const uint8_t* base = q16LUT + col * 256;
                q16LUTr[col*4+0] = _mm512_loadu_si512((const void*)(base +   0));
                q16LUTr[col*4+1] = _mm512_loadu_si512((const void*)(base +  64));
                q16LUTr[col*4+2] = _mm512_loadu_si512((const void*)(base + 128));
                q16LUTr[col*4+3] = _mm512_loadu_si512((const void*)(base + 192));
            }
        }
        // transpose(numCentroids, numSubspaces, lut);
        if constexpr(timing) {
            auto t1 = clock::now();
            lutLatency += std::chrono::duration_cast<std::chrono::nanoseconds>(t1 - t0).count();
        }


        return;
    }


    std::vector<ID<dist_t>> searchKNNPQ(
        const void *query, const void *oQuery,
        const int efSearch, const int k, const int numRefine
    ) {
        MaxHeap<qdist_t> topCandidates;
        MinHeap<qdist_t> candidateSet;

        int ef = std::max(efSearch, k);
        assert(numRefine >= k);
        traversal = 0;

        computeLUT(static_cast<const dist_t*>(query));

        id_t curVecID = entryPoint;
        qdist_t curDist = std::numeric_limits<qdist_t>::max();

        // top level init
        if (maxLevel > 0 && maxLevelElementsCt > 1) {
            alignas(64) qdist_t dists[batchSize];
            const auto& curEdge = nodes[entryPoint].edges[maxLevel - 1];
            search8BitAVX512<numSubspaces, true, batchSize>(q8LUTr, curEdge.neighborsCode, dists);
            for (int i = 0; i < maxLevelElementsCt; ++i) {
                if (curDist > dists[i]) {
                    curDist = dists[i];
                    curVecID = curEdge.neighbors[i];
                }
            }
        }

        // greedy descent on levels maxLevel-1 ... 1
        for (int level = (int)maxLevel - 1; level > 0; --level) {
            bool changed = true;
            while (changed) {
                changed = false;
                const auto& curEdge = nodes[curVecID].edges[level - 1];
                alignas(64) qdist_t dists[batchSize];
                search8BitAVX512<numSubspaces, true, batchSize>(q8LUTr, curEdge.neighborsCode, dists);
                for (int i = 0; i < batchSize; ++i) {
                    id_t nextVecID = curEdge.neighbors[i];
                    qdist_t tempDist = dists[i];
                    if (tempDist < curDist) {
                        curDist = tempDist;
                        curVecID = nextVecID;
                        changed = true;
                    }
                }
            }
        }

        visitedArray.nextGeneration();
        qdist_t lowerBound = std::numeric_limits<qdist_t>::max();

        std::array<id_t, level0NodeBatchSize> curVecIDs;

#ifdef MULTI_START

        // --- Opt ----
        size_t seedCount = 0;

        // 默认先放当前 greedy 到的入口
        curVecIDs[seedCount++] = curVecID;

        // 如果存在 level 1，则从 level 1 邻居里再挑一些
        if (maxLevel >= 1 && !nodes[curVecID].edges.empty()) {
            const auto& lastEdge = nodes[curVecID].edges[0];  // 真实 level 1
            alignas(64) qdist_t dists[batchSize];
            search8BitAVX512<numSubspaces, true, batchSize>(q8LUTr, lastEdge.neighborsCode, dists);

            std::vector<std::pair<qdist_t, id_t>> candidates;
            candidates.reserve(batchSize + 1);

            // 把 curVecID 自己也放进去
            candidates.emplace_back(curDist, curVecID);

            for (int i = 0; i < batchSize; ++i) {
                id_t nid = lastEdge.neighbors[i];
                candidates.emplace_back(dists[i], nid);
            }

            std::sort(candidates.begin(), candidates.end(),
                    [](const auto& a, const auto& b) { return a.first < b.first; });

            std::unordered_set<id_t> used;
            for (const auto& [dist, vid] : candidates) {
                if (used.insert(vid).second) {
                    curVecIDs[used.size() - 1] = vid;
                    if (used.size() == level0NodeBatchSize) break;
                }
            }
            seedCount = used.size();
        }

        // 不足的部分补当前入口
        for (size_t i = seedCount; i < level0NodeBatchSize; ++i) {
            curVecIDs[i] = curVecID;
        }
        for (size_t i = 0; i < seedCount; ++i) {
            topCandidates.emplace(curVecIDs[i], lowerBound);
            candidateSet.push(ID<qdist_t>(curVecIDs[i], lowerBound));
        }
        // --- Opt ----
#else
        curVecIDs[0] = curVecID;
        for (size_t i = 1; i < level0NodeBatchSize; ++i) {
            curVecIDs[i] = curVecID;
        }

        for (int i = 0; i < level0NodeBatchSize; ++i) {
            topCandidates.emplace(curVecIDs[i], lowerBound);
            candidateSet.emplace(curVecIDs[i], lowerBound);
        }
#endif

        while (!candidateSet.empty()) {
            std::array<ID<qdist_t>, level0NodeBatchSize> curIDs;
            int actualNodeBatchSize = 0;

            while (!candidateSet.empty() && actualNodeBatchSize < level0NodeBatchSize) {
                auto x = candidateSet.top();
                candidateSet.pop();
                if (visitedArray.isVisited(x.vecID)) continue;
                visitedArray.markVisited(x.vecID);
                curIDs[actualNodeBatchSize++] = x;
            }

            if (actualNodeBatchSize == 0) continue;

            if (curIDs[0].dist > lowerBound && topCandidates.size() == (size_t)ef) {
                break;
            }

            if constexpr (timing) {
                traversal++;
            }

            alignas(64) qdist_t dists[batchSize];

            if constexpr (level0NodeBatchSize == 1) {
                // numLevel0Edges == batchSize（no-CA 消融）：单节点已占满整批且布局一致，无需拼接拷贝
                search8BitAVX512<numSubspaces, true, batchSize>(
                    q8LUTr, nodes[curIDs[0].vecID].level0Edges.neighborsCode, dists);
            } else {
                // 把多个 level0 block 重新按 subspace-major 拼成 batchSize 宽
                alignas(64) code_t neighborsCode[numSubspaces * batchSize];
                std::fill_n(neighborsCode, numSubspaces * batchSize, 0);

                for (size_t s = 0; s < numSubspaces; ++s) {
                    code_t* dst = neighborsCode + s * batchSize;

                    for (int nodeID = 0; nodeID < actualNodeBatchSize; ++nodeID) {
                        id_t vid = curIDs[nodeID].vecID;
                        const code_t* src =
                            nodes[vid].level0Edges.neighborsCode + s * numLevel0Edges;

                        std::copy_n(src, numLevel0Edges, dst + nodeID * numLevel0Edges);
                    }
                }

                search8BitAVX512<numSubspaces, true, batchSize>(q8LUTr, neighborsCode, dists);
            }

            for (int nodeID = 0; nodeID < actualNodeBatchSize; ++nodeID) {
                auto &curEdge = nodes[curIDs[nodeID].vecID].level0Edges;
                qdist_t *curDistPtr = dists + nodeID * numLevel0Edges;

                for (int edgeID = 0; edgeID < numLevel0Edges; ++edgeID) {
                    qdist_t nextDist = curDistPtr[edgeID];
                    id_t nextVecID = curEdge.neighbors[edgeID];

                    if (topCandidates.size() < (size_t)ef || lowerBound > nextDist) {
                        candidateSet.emplace(nextVecID, nextDist);
                        topCandidates.emplace(nextVecID, nextDist);
                        if (topCandidates.size() > (size_t)ef)
                            topCandidates.pop();
                        if (!topCandidates.empty())
                            lowerBound = topCandidates.top().dist;
                    }
                }
            }
        }

        std::vector<ID<dist_t>> results;
        if (numRefine > k) {
            while (topCandidates.size() > (size_t)numRefine) {
                topCandidates.pop();
            }

            size_t dim_ = dim;
            while (!topCandidates.empty()) {
                id_t vid = topCandidates.top().vecID;
                dist_t curDist = distFunc(
                    static_cast<const void*>(rawVectors.get() + vid * dim),
                    oQuery,
                    &dim_
                );
                results.emplace_back(vid, curDist);
                topCandidates.pop();
            }

            std::partial_sort(results.begin(), results.begin() + k, results.end());
            results.resize(k);
        } else {
            while (topCandidates.size() > (size_t)k) topCandidates.pop();
            while (!topCandidates.empty()) {
                const auto &curID = topCandidates.top();
                results.emplace_back(curID.vecID, curID.dist);
                topCandidates.pop();
            }
        }

        for (ID<dist_t> &curID : results) {
            curID.vecID = toExternalID[curID.vecID];
        }

        return results;
    }


    std::vector<ID<dist_t>> searchKNNPQ16(
        const void *query, const void *oQuery,
        const int efSearch, const int k, const int numRefine
    ) {
        MaxHeap<qdist16_t> topCandidates;
        MinHeapFast<ID<qdist16_t>> candidateSet;

        int ef = std::max(efSearch, k);
        assert(numRefine >= k);
        traversal = 0;

        computeLUT(static_cast<const dist_t*>(query), true);

        id_t curVecID = entryPoint;
        qdist16_t curDist = std::numeric_limits<qdist16_t>::max();

        // top level init
        if (maxLevel > 0 && maxLevelElementsCt > 1) {
            alignas(64) qdist16_t dists[batchSize];
            const auto& curEdge = nodes[entryPoint].edges[maxLevel - 1];
            search8BitAVX512_u16sum<numSubspaces, timing, batchSize>(q16LUTr, curEdge.neighborsCode, dists);
            for (int i = 0; i < maxLevelElementsCt; ++i) {
                if (curDist > dists[i]) {
                    curDist = dists[i];
                    curVecID = curEdge.neighbors[i];
                }
            }
        }

        // greedy descent on levels maxLevel-1 ... 1
        for (int level = (int)maxLevel - 1; level > 0; --level) {
            bool changed = true;
            while (changed) {
                changed = false;
                const auto& curEdge = nodes[curVecID].edges[level - 1];
                alignas(64) qdist16_t dists[batchSize];
                search8BitAVX512_u16sum<numSubspaces, timing, batchSize>(q16LUTr, curEdge.neighborsCode, dists);
                for (int i = 0; i < batchSize; ++i) {
                    id_t nextVecID = curEdge.neighbors[i];
                    qdist16_t tempDist = dists[i];
                    if (tempDist < curDist) {
                        curDist = tempDist;
                        curVecID = nextVecID;
                        changed = true;
                    }
                }
            }
        }

        visitedArray.nextGeneration();
        qdist16_t lowerBound = std::numeric_limits<qdist16_t>::max();

        // 关键：level0 batching 退化时，seed 应该是 curVecID 本身
        std::array<id_t, level0NodeBatchSize> curVecIDs;

#ifdef MULTI_START

        // --- Opt ----
        size_t seedCount = 0;

        // 默认先放当前 greedy 到的入口
        curVecIDs[seedCount++] = curVecID;

        // 如果存在 level 1，则从 level 1 邻居里再挑一些
        if (maxLevel >= 1 && !nodes[curVecID].edges.empty()) {
            const auto& lastEdge = nodes[curVecID].edges[0];  // 真实 level 1
            alignas(64) qdist16_t dists[batchSize];
            search8BitAVX512_u16sum<numSubspaces, timing, batchSize>(q16LUTr, lastEdge.neighborsCode, dists);

            std::vector<std::pair<qdist16_t, id_t>> candidates;
            candidates.reserve(batchSize + 1);

            // 把 curVecID 自己也放进去
            candidates.emplace_back(curDist, curVecID);

            for (int i = 0; i < batchSize; ++i) {
                id_t nid = lastEdge.neighbors[i];
                candidates.emplace_back(dists[i], nid);
            }

            std::sort(candidates.begin(), candidates.end(),
                    [](const auto& a, const auto& b) { return a.first < b.first; });

            std::unordered_set<id_t> used;
            for (const auto& [dist, vid] : candidates) {
                if (used.insert(vid).second) {
                    curVecIDs[used.size() - 1] = vid;
                    if (used.size() == level0NodeBatchSize) break;
                }
            }
            seedCount = used.size();
        }

        // 不足的部分补当前入口
        for (size_t i = seedCount; i < level0NodeBatchSize; ++i) {
            curVecIDs[i] = curVecID;
        }
        for (size_t i = 0; i < seedCount; ++i) {
            topCandidates.emplace(curVecIDs[i], lowerBound);
            candidateSet.push(ID<qdist16_t>(curVecIDs[i], lowerBound));
        }
        // --- Opt ----
#else

        curVecIDs[0] = curVecID;
        for (size_t i = 1; i < level0NodeBatchSize; ++i) {
            curVecIDs[i] = curVecID;
        }
        for (int i = 0; i < level0NodeBatchSize; ++i) {
            topCandidates.emplace(curVecIDs[i], lowerBound);
            candidateSet.push(ID<qdist16_t>(curVecIDs[i], lowerBound));
        }
#endif

        while (!candidateSet.empty()) {
            std::array<ID<qdist16_t>, level0NodeBatchSize> curIDs;
            int actualNodeBatchSize = 0;

            while (!candidateSet.empty() && actualNodeBatchSize < level0NodeBatchSize) {
                auto x = candidateSet.top();
                candidateSet.pop();
                if (visitedArray.isVisited(x.vecID)) continue;
                visitedArray.markVisited(x.vecID);
                curIDs[actualNodeBatchSize++] = x;
            }

            if (actualNodeBatchSize == 0) continue;

            if (curIDs[0].dist > lowerBound && topCandidates.size() == (size_t)ef) {
                break;
            }

            if constexpr (timing) {
                traversal++;
            }

            if (!candidateSet.empty()) {
                id_t prefetchID = candidateSet.next_id();
                mem_prefetch_l2(nodes[prefetchID].level0Edges.neighborsCode, 20);
            }

            alignas(64) qdist16_t dists[batchSize];

            if constexpr (level0NodeBatchSize == 1) {
                // numLevel0Edges == batchSize（no-CA 消融）：单节点已占满整批且布局一致，无需拼接拷贝
                search8BitAVX512_u16sum<numSubspaces, timing, batchSize>(
                    q16LUTr, nodes[curIDs[0].vecID].level0Edges.neighborsCode, dists);
            } else {
                alignas(64) code_t neighborsCode[numSubspaces * batchSize];
                std::fill_n(neighborsCode, numSubspaces * batchSize, 0);
                for (size_t s = 0; s < numSubspaces; ++s) {
                    code_t* dst = neighborsCode + s * batchSize;

                    for (int nodeID = 0; nodeID < actualNodeBatchSize; ++nodeID) {
                        id_t vid = curIDs[nodeID].vecID;
                        const code_t* src = nodes[vid].level0Edges.neighborsCode + s * numLevel0Edges;

                        std::copy_n(src, numLevel0Edges, dst + nodeID * numLevel0Edges);
                    }
                }

                search8BitAVX512_u16sum<numSubspaces, timing, batchSize>(q16LUTr, neighborsCode, dists);
            }

            for (int nodeID = 0; nodeID < actualNodeBatchSize; ++nodeID) {
                auto &curEdge = nodes[curIDs[nodeID].vecID].level0Edges;
                qdist16_t *curDistPtr = dists + nodeID * numLevel0Edges;

                for (int edgeID = 0; edgeID < numLevel0Edges; ++edgeID) {
                    qdist16_t nextDist = curDistPtr[edgeID];
                    id_t nextVecID = curEdge.neighbors[edgeID];

                    if (topCandidates.size() < (size_t)ef || lowerBound > nextDist) {
                        topCandidates.emplace(nextVecID, nextDist);
                        candidateSet.push(ID<qdist16_t>(nextVecID, nextDist));
                        if (topCandidates.size() > (size_t)ef)
                            topCandidates.pop();
                        if (!topCandidates.empty())
                            lowerBound = topCandidates.top().dist;
                    }
                }
            }
        }

        std::vector<ID<dist_t>> results;
        if (numRefine > k) {
            while (topCandidates.size() > (size_t)numRefine) {
                topCandidates.pop();
            }

            size_t dim_ = dim;
            while (!topCandidates.empty()) {
                id_t vid = topCandidates.top().vecID;
                dist_t curDist = distFunc(
                    static_cast<const void*>(rawVectors.get() + vid * dim),
                    oQuery,
                    &dim_
                );
                results.emplace_back(vid, curDist);
                topCandidates.pop();
            }

            std::partial_sort(results.begin(), results.begin() + k, results.end());
            results.resize(k);
        } else {
            while (topCandidates.size() > (size_t)k) topCandidates.pop();
            while (!topCandidates.empty()) {
                const auto &curID = topCandidates.top();
                results.emplace_back(curID.vecID, curID.dist);
                topCandidates.pop();
            }
        }

        return results;
    }

    std::vector<ID<dist_t>> searchKNNPQ16NoCA(const void *query, const void *oQuery, const int efSearch,
        const int k, const int numRefine
    ) {
        // MaxHeap<qdist16_t> topCandidates;
        // MinHeap<qdist16_t> candidateSet;
        //FixedKHeap<15, ID<qdist16_t>, true> topCandidates;
        MaxHeap<qdist16_t> topCandidates;
        MinHeapFast<ID<qdist16_t>> candidateSet;

        int ef = std::max(efSearch, k);
        assert(numRefine >= k);
        size_t dim_ = dim;
        traversal = 0;

        computeLUT(static_cast<const dist_t*>(query), true);

        id_t curVecID = entryPoint;
        qdist16_t curDist = std::numeric_limits<qdist16_t>::max();

        if (maxLevel > 0 && maxLevelElementsCt > 1) {
            alignas(64) qdist16_t dists[batchSize];
            const auto& curEdge = nodes[entryPoint].edges[maxLevel - 1];
            search8BitAVX512_u16sum<numSubspaces, timing, batchSize>(q16LUTr, curEdge.neighborsCode, dists);
            for (int i = 0; i < maxLevelElementsCt; ++i) {
                if (curDist > dists[i]) {
                    curDist = dists[i];
                    curVecID = curEdge.neighbors[i];
                }
            }
        }

        for (int level = maxLevel - 1; level > 0; --level) {
            bool changed = true;
            while (changed) {
                changed = false;
                const auto& curEdge = nodes[curVecID].edges[level - 1];
                alignas(64) qdist16_t dists[batchSize];
                search8BitAVX512_u16sum<numSubspaces, timing, batchSize>(q16LUTr, curEdge.neighborsCode, dists);
                for (int i = 0; i < batchSize; ++i) {
                    id_t nextVecID = curEdge.neighbors[i];
                    qdist16_t tempDist = dists[i];
                    if (tempDist < curDist) {
                        curDist = tempDist;
                        curVecID = nextVecID;
                        changed = true;
                    }
                }
            }
        }

        visitedArray.nextGeneration();
        qdist16_t lowerBound = std::numeric_limits<qdist16_t>::max();
        
        topCandidates.emplace(curVecID, lowerBound);
        // candidateSet.emplace(curVecID, lowerBound);
        // topCandidates.try_push(ID<qdist16_t>(curVecID, lowerBound));
        candidateSet.push(ID<qdist16_t>(curVecID, lowerBound));


        while(!candidateSet.empty()) {
            ID<qdist16_t> curID = candidateSet.top();
            if ((curID.dist) > lowerBound && topCandidates.size() == ef) {
                break;
            }

            candidateSet.pop();

            if constexpr(timing) {
                traversal++;
            }

            if(visitedArray.isVisited(curID.vecID)) continue;
            visitedArray.markVisited(curID.vecID);

            id_t prefetchID = candidateSet.next_id();
            mem_prefetch_l2(nodes[prefetchID].edges[0].neighborsCode, 20);

            id_t curVecID = curID.vecID;
            const auto&curEdge = nodes[curVecID].edges[0];

            alignas (64) qdist16_t dists[batchSize];
            search8BitAVX512_u16sum<numSubspaces, timing, batchSize>(q16LUTr, curEdge.neighborsCode, dists);
            
            size_t numN = this->numNeighbors[0][curVecID];
            for(int edgeID = 0;edgeID < numN; ++edgeID) {
                id_t nextVecID = curEdge.neighbors[edgeID];

                qdist16_t nextDist = dists[edgeID];

                if (topCandidates.size() < ef || lowerBound > nextDist) {
                    // candidateSet.emplace(nextVecID, nextDist);
                    topCandidates.emplace(nextVecID, nextDist);
                    // topCandidates.try_push(ID<qdist16_t>(nextVecID, nextDist));
                    candidateSet.push(ID<qdist16_t>(nextVecID, nextDist));
                    if (topCandidates.size() > ef)
                        topCandidates.pop();
                    if (!topCandidates.empty())
                        lowerBound = topCandidates.top().dist;
                }

            }

        }

        // printf("Final lb is %d\n", int(lowerBound));

        std::vector<ID<dist_t>> results;
        if(numRefine > k) {
            while(topCandidates.size() > numRefine) {
                topCandidates.pop();
            }

            size_t dim_ = dim;
            while(topCandidates.size() > 0) {
                id_t curVecID = topCandidates.top().vecID;
                dist_t curDist = distFunc(
                    static_cast<const void*>((rawVectors.get()) + curVecID * dim),
                    oQuery,
                    &dim_
                );
                results.emplace_back(curVecID, curDist);
                topCandidates.pop();
            }

            std::partial_sort(results.begin(), results.begin()+k, results.end());
            results.resize(k);
        } else {
            while(topCandidates.size() > k) topCandidates.pop();
            while(topCandidates.size() > 0) {
                const ID<qdist16_t> &curID = topCandidates.top();
                results.emplace_back(curID.vecID, curID.dist);
                topCandidates.pop();
            }
        }

        // for(ID<dist_t> &curID : results) {
        //     curID.vecID = toExternalID[curID.vecID];
        // }

        return results;
    }

    std::vector<ID<dist_t>> searchKNNPQ16Batch(const void *query, const void *oQuery, const int efSearch,
        const int k, const int numRefine
    ) {

        assert(batchSize == 32);

        MaxHeap<qdist16_t> topCandidates;
        MinHeapFast<ID<qdist16_t>> candidateSet;

        int ef = std::max(efSearch, k);
        assert(numRefine >= k);
        size_t dim_ = dim;

        computeLUT(static_cast<const dist_t*>(query), true);

        id_t curVecID = entryPoint;
        qdist16_t curDist = std::numeric_limits<qdist16_t>::max();
        alignas(64) qdist16_t dists[64];
        const auto& curEdge = nodes[entryPoint].edges[maxLevel];
        alignas(64) uint8_t emptyCodes[batchSize*numSubspaces] = {}; // All 0
        search8BitAVX512_u16sumBatch<numSubspaces, timing>(q16LUTr, curEdge.neighborsCode, emptyCodes, dists);
        assert(maxLevelElementsCt <= batchSize);
        for(int i = 0; i < maxLevelElementsCt; ++i) {
            if(curDist > dists[i]) {
                curDist = dists[i];
                curVecID = curEdge.neighbors[i];
            }
        }

        for (int level = maxLevel - 1; level > 0; level--) {
            bool changed = true;
            while(changed) {
                changed = false;
                const auto& curEdge = nodes[curVecID].edges[level];
                alignas(64) qdist16_t dists[64];
                search8BitAVX512_u16sumBatch<numSubspaces, timing>(q16LUTr, curEdge.neighborsCode, emptyCodes, dists);
                for(int i=0;i<batchSize;++i) {
                    id_t nextVecID = curEdge.neighbors[i];
                    qdist16_t tempDist = dists[i];
                    if(tempDist < curDist) {
                        curDist = tempDist;
                        curVecID = nextVecID;
                        changed = true;
                    }
                }
            }
        }

        visitedArray.nextGeneration();
        qdist16_t lowerBound = std::numeric_limits<qdist16_t>::max();
        
        topCandidates.emplace(curVecID, lowerBound);
        // candidateSet.emplace(curVecID, lowerBound);
        // topCandidates.try_push(ID<qdist16_t>(curVecID, lowerBound));
        candidateSet.push(ID<qdist16_t>(curVecID, lowerBound));


        while (!candidateSet.empty()) {
            if (candidateSet.size() < 2) {
                // 不足两个时单独处理一个，兼容原逻辑
                ID<qdist16_t> curID = candidateSet.top();
                candidateSet.pop();

                if (visitedArray.isVisited(curID.vecID)) continue;
                visitedArray.markVisited(curID.vecID);

                id_t curVecID = curID.vecID;
                const auto& curEdge = nodes[curVecID].edges[0];

                alignas(64) qdist16_t dists[batchSize*2];
                search8BitAVX512_u16sumBatch<numSubspaces, timing>(
                    q16LUTr, curEdge.neighborsCode, emptyCodes, dists);

                for (int edgeID = 0; edgeID < batchSize; ++edgeID) {
                    id_t nextVecID = curEdge.neighbors[edgeID];
                    qdist16_t nextDist = dists[edgeID];
                    if (topCandidates.size() < ef || lowerBound > nextDist) {
                        candidateSet.push(ID<qdist16_t>(nextVecID, nextDist));
                        topCandidates.emplace(nextVecID, nextDist);
                        if (topCandidates.size() > ef) topCandidates.pop();
                        lowerBound = topCandidates.top().dist;
                    }
                }
                continue;
            }

            // ---------- 批处理两个 ----------
            ID<qdist16_t> curID1 = candidateSet.top(); candidateSet.pop();
            ID<qdist16_t> curID2 = candidateSet.top(); candidateSet.pop();

            if (visitedArray.isVisited(curID1.vecID)) continue;
            if (visitedArray.isVisited(curID2.vecID)) continue;

            visitedArray.markVisited(curID1.vecID);
            visitedArray.markVisited(curID2.vecID);

            id_t prefetchID1 = candidateSet.next_id();
            mem_prefetch_l2(nodes[prefetchID1].edges[0].neighborsCode, 20);
            // id_t prefetchID2 = candidateSet.next_next_id();
            // mem_prefetch_l2(nodes[prefetchID2].edges[0].neighborsCode, 20);

            const auto& edge1 = nodes[curID1.vecID].edges[0];
            const auto& edge2 = nodes[curID2.vecID].edges[0];

            // 一次批处理 64 距离
            alignas(64) qdist16_t dists[batchSize * 2];
            search8BitAVX512_u16sumBatch<numSubspaces, timing>(
                q16LUTr, edge1.neighborsCode, edge2.neighborsCode, dists);

            // 批1
            for (int edgeID = 0; edgeID < batchSize; ++edgeID) {
                id_t nextVecID = edge1.neighbors[edgeID];
                qdist16_t nextDist = dists[edgeID];
                if (topCandidates.size() < ef || lowerBound > nextDist) {
                    candidateSet.push(ID<qdist16_t>(nextVecID, nextDist));
                    topCandidates.emplace(nextVecID, nextDist);
                    if (topCandidates.size() > ef) topCandidates.pop();
                    lowerBound = topCandidates.top().dist;
                }
            }

            // 批2
            for (int edgeID = 0; edgeID < batchSize; ++edgeID) {
                id_t nextVecID = edge2.neighbors[edgeID];
                qdist16_t nextDist = dists[batchSize + edgeID];
                if (topCandidates.size() < ef || lowerBound > nextDist) {
                    candidateSet.push(ID<qdist16_t>(nextVecID, nextDist));
                    topCandidates.emplace(nextVecID, nextDist);
                    if (topCandidates.size() > ef) topCandidates.pop();
                    lowerBound = topCandidates.top().dist;
                }
            }
        }

        // printf("Final lb is %d\n", int(lowerBound));

        std::vector<ID<dist_t>> results;
        if(numRefine > k) {
            while(topCandidates.size() > numRefine) {
                topCandidates.pop();
            }

            size_t dim_ = dim;
            while(topCandidates.size() > 0) {
                id_t curVecID = topCandidates.top().vecID;
                dist_t curDist = distFunc(
                    static_cast<const void*>((rawVectors.get()) + curVecID * dim),
                    oQuery,
                    &dim_
                );
                results.emplace_back(curVecID, curDist);
                topCandidates.pop();
            }

            std::partial_sort(results.begin(), results.begin()+k, results.end());
            results.resize(k);
        } else {
            while(topCandidates.size() > k) topCandidates.pop();
            while(topCandidates.size() > 0) {
                const ID<qdist16_t> &curID = topCandidates.top();
                results.emplace_back(curID.vecID, curID.dist);
                topCandidates.pop();
            }
        }

        // for(ID<dist_t> &curID : results) {
        //     curID.vecID = toExternalID[curID.vecID];
        // }

        return results;
    }


    std::vector<ID<dist_t>> searchKNNPQR(const void *query, const void *oQuery, const int efSearch,
        const int k
    ) {
        int ef = std::max(efSearch, k);
        size_t dim_ = dim;

        computeLUT(static_cast<const dist_t*>(query));

        id_t curVecID = entryPoint;
        qdist_t curDist = std::numeric_limits<qdist_t>::max();
        alignas(64) qdist_t dists[batchSize];
        const auto& curEdge = nodes[entryPoint].edges[maxLevel];
        search8BitAVX512<numSubspaces, true, batchSize>(q8LUTr, curEdge.neighborsCode, dists);
        for(int i = 0; i < maxLevelElementsCt; ++i) {
            if(curDist > dists[i]) {
                curDist = dists[i];
                curVecID = curEdge.neighbors[i];
            }
        }
        

        for (int level = maxLevel - 1; level > 0; level--) {
            bool changed = true;
            while(changed) {
                changed = false;
                const auto&curEdge = nodes[curVecID].edges[level];
                alignas(64) qdist_t dists[batchSize];
                search8BitAVX512<numSubspaces, true, batchSize>(q8LUTr, curEdge.neighborsCode, dists);
                for(int i=0;i<batchSize;++i) {
                    id_t nextVecID = curEdge.neighbors[i];
                    qdist_t tempDist = dists[i];
                    if(tempDist < curDist) {
                        curDist = tempDist;
                        curVecID = nextVecID;
                        changed = true;
                    }
                }
            }
        }

        visitedArray.nextGeneration();
        dist_t lowerBound = std::numeric_limits<dist_t>::max();
        MaxHeap<dist_t> topCandidates;
        MinHeap<dist_t> candidateSet;
        std::priority_queue<qdist_t, std::vector<qdist_t>, std::less<qdist_t>> lbBounds;
        topCandidates.emplace(curVecID, lowerBound);
        candidateSet.emplace(curVecID, lowerBound);
        lbBounds.push(std::numeric_limits<qdist_t>::max());

        while(!candidateSet.empty()) {
            ID<dist_t> curID = candidateSet.top();
            if ((curID.dist) > lowerBound && topCandidates.size() == ef) {
                break;
            }
            candidateSet.pop();

            id_t curVecID = curID.vecID;
            const auto&curEdge = nodes[curVecID].edges[0];

            alignas (64) qdist_t dists[batchSize];
            search8BitAVX512<numSubspaces, true, batchSize>(q8LUTr, curEdge.neighborsCode, dists);

            for(int edgeID = 0;edgeID < batchSize; ++edgeID) {
                id_t nextVecID = curEdge.neighbors[edgeID];

                if(visitedArray.isVisited(nextVecID)) continue;
                visitedArray.markVisited(nextVecID);

                qdist_t nextDist = dists[edgeID];

                // dist_t distsF[batchSize];
                // computePQDistance(lut, codes, distsF);

                if (topCandidates.size() < ef || lbBounds.top() > nextDist) {
                    dist_t nextDistF = distFunc(
                        oQuery,
                        static_cast<const void*>(rawVectors.get() + (nextVecID * dim)),
                        &dim_
                    );


                    if(lowerBound > nextDistF){
                        candidateSet.emplace(nextVecID, nextDistF);
                        topCandidates.emplace(nextVecID, nextDistF);
                        lbBounds.push(nextDist);
                        if (topCandidates.size() > ef) {
                            topCandidates.pop();
                        }
                        if (!topCandidates.empty()){
                            lowerBound = topCandidates.top().dist;

                        }
                    }
                }

            }

        }

        // printf("Final lb is %d\n", int(lowerBound));

        std::vector<ID<dist_t>> results;
        while(topCandidates.size() > k) topCandidates.pop();
        while(topCandidates.size() > 0) {
            const ID<dist_t> &curID = topCandidates.top();
            results.emplace_back(curID.vecID, curID.dist);
            topCandidates.pop();
        }

        for(ID<dist_t> &curID : results) {
            curID.vecID = toExternalID[curID.vecID];
        }

        return results;
    }

    
    std::vector<ID<dist_t>> searchKNN(const void *oQuery, const int efSearch, const int k
    ) {
        MaxHeap<dist_t> topCandidates;
        MinHeap<dist_t> candidateSet;
        int ef = std::max(efSearch, k);
        assert(numRefine >= k);
        traversal = 0;

        size_t dim_ = dim;

        id_t curVecID = entryPoint;
        dist_t curDist = std::numeric_limits<qdist_t>::max();
        const auto& curEdge = nodes[entryPoint].edges[maxLevel];
        for(int i = 0; i < maxLevelElementsCt; ++i) {
            id_t nextID = curEdge.neighbors[i];
            dist_t nextDist = distFunc(
                oQuery,
                static_cast<const void*>(rawVectors.get() + (nextID * dim)),
                &dim_
            );
            if(curDist > nextDist) {
                curDist = nextDist;
                curVecID = nextID;
            }
        }
        

        for (int level = maxLevel - 1; level > 0; level--) {
            bool changed = true;
            while(changed) {
                changed = false;
                const auto& curEdge = nodes[curVecID].edges[level];
                alignas(64) dist_t dists[batchSize];
                for(int idx = 0; idx < batchSize; ++idx) {
                    const id_t& curID = curEdge.neighbors[idx];
                    dists[idx] = distFunc(
                        oQuery,
                        static_cast<const void*>(rawVectors.get() + (curID * dim)),
                        &dim_
                    );
                }
                for(int i=0;i<batchSize;++i) {
                    id_t nextVecID = curEdge.neighbors[i];
                    dist_t tempDist = dists[i];
                    if(tempDist < curDist) {
                        curDist = tempDist;
                        curVecID = nextVecID;
                        changed = true;
                    }
                }
            }
        }

        visitedArray.nextGeneration();
        dist_t lowerBound = std::numeric_limits<dist_t>::max();
        
        topCandidates.emplace(curVecID, lowerBound);
        candidateSet.emplace(curVecID, lowerBound);

        while(!candidateSet.empty()) {
            ID<dist_t> curID = candidateSet.top();
            if ((curID.dist) > lowerBound && topCandidates.size() == ef) {
                break;
            }
            candidateSet.pop();

            if constexpr (timing) {
                traversal++;
            }

            id_t curVecID = curID.vecID;
            const auto&curEdge = nodes[curVecID].edges[0];

            alignas (64) dist_t dists[batchSize];
            for(int idx = 0; idx < batchSize; ++idx) {
                const id_t& curID = curEdge.neighbors[idx];
                dists[idx] = distFunc(
                    oQuery,
                    static_cast<const void*>(rawVectors.get() + (curID * dim)),
                    &dim_
                );
            }

            for(int edgeID = 0;edgeID < batchSize; ++edgeID) {
                id_t nextVecID = curEdge.neighbors[edgeID];

                if(visitedArray.isVisited(nextVecID)) continue;
                visitedArray.markVisited(nextVecID);

                dist_t nextDist = dists[edgeID];

                if (topCandidates.size() < ef || lowerBound > nextDist) {
                    candidateSet.emplace(nextVecID, nextDist);
                    topCandidates.emplace(nextVecID, nextDist);
                    if (topCandidates.size() > ef)
                        topCandidates.pop();
                    if (!topCandidates.empty())
                        lowerBound = topCandidates.top().dist;
                }

            }

        }

        // printf("Final lb is %f\n", lowerBound);

        std::vector<ID<dist_t>> results;
        while(topCandidates.size() > k) topCandidates.pop();
        while(topCandidates.size() > 0) {
            const ID<dist_t> &curID = topCandidates.top();
            results.emplace_back(curID.vecID, curID.dist);
            topCandidates.pop();
        }

        for(ID<dist_t> &curID : results) {
            curID.vecID = toExternalID[curID.vecID];
        }

        return results;
    }

    MaxHeap<dist_t> searchKNNForAdding(
        const void *oQuery,
        const int level,                 // 真实level: 0,1,2,...
        const id_t entryPoint,
        const size_t maxElements
    ) {
        MaxHeap<dist_t> topCandidates;
        MinHeap<dist_t> candidateSet;

        const int ef = efConstruction;
        size_t dim_ = dim;

        visitedArray.nextGeneration();

        dist_t lowerBound = distFunc(
            static_cast<const void*>(rawVectors.get() + (entryPoint * dim)),
            oQuery,
            &dim_
        );

        topCandidates.emplace(entryPoint, lowerBound);
        candidateSet.emplace(entryPoint, lowerBound);
        visitedArray.markVisited(entryPoint);

        while (!candidateSet.empty()) {

            ID<dist_t> curID = candidateSet.top();
            if (curID.dist > lowerBound && topCandidates.size() == (size_t)ef) break;
            candidateSet.pop();

            const id_t curVecID = curID.vecID;

            // 构建阶段你用了锁：保留（读 edges + 读 numNeighbors）
            std::unique_lock<std::mutex> lock(this->nodesLock.at(curVecID));

            const size_t deg = numNeighbors.at(level).at(curVecID);
            if (deg == 0) continue;

            if (level == 0) {
                // -------- L0: level0Edges --------
                const auto &curEdge = nodes.at(curVecID).level0Edges;

                for (size_t edgeID = 0; edgeID < deg; ++edgeID) {
                    const id_t nextVecID = curEdge.neighbors[edgeID];
                    // if ((size_t)nextVecID >= curElements) continue;
                    if ((size_t)nextVecID >= maxElements) continue;

                    if (visitedArray.isVisited(nextVecID)) continue;
                    visitedArray.markVisited(nextVecID);

                    dist_t nextDist = distFunc(
                        static_cast<const void*>(rawVectors.get() + dim * nextVecID),
                        oQuery,
                        &dim_
                    );

                    if (topCandidates.size() < (size_t)ef || lowerBound > nextDist) {
                        candidateSet.emplace(nextVecID, nextDist);
                        topCandidates.emplace(nextVecID, nextDist);

                        if (topCandidates.size() > (size_t)ef)
                            topCandidates.pop();
                        if (!topCandidates.empty())
                            lowerBound = topCandidates.top().dist;
                    }
                }

            } else {
                // -------- L>0: edges[level-1] --------
                // 注意：level-1 必须小于 edges.size()
                if ((size_t)(level - 1) >= nodes.at(curVecID).edges.size()) continue;

                const auto &curEdge = nodes.at(curVecID).edges.at(level - 1);

                for (size_t edgeID = 0; edgeID < deg; ++edgeID) {
                    const id_t nextVecID = curEdge.neighbors[edgeID];
                    //if ((size_t)nextVecID >= curElements) continue;
                    if ((size_t)nextVecID >= maxElements) continue;

                    if (visitedArray.isVisited(nextVecID)) continue;
                    visitedArray.markVisited(nextVecID);

                    dist_t nextDist = distFunc(
                        static_cast<const void*>(rawVectors.get() + dim * nextVecID),
                        oQuery,
                        &dim_
                    );

                    if (topCandidates.size() < (size_t)ef || lowerBound > nextDist) {
                        candidateSet.emplace(nextVecID, nextDist);
                        topCandidates.emplace(nextVecID, nextDist);

                        if (topCandidates.size() > (size_t)ef)
                            topCandidates.pop();
                        if (!topCandidates.empty())
                            lowerBound = topCandidates.top().dist;
                    }
                }
            }
        }

        return topCandidates;
    }

    // Given a vector of neighbors candidates, we want to choose some of them that close to the new small node but diverse enough
    void getNeighborsByHeuristic2(
        MaxHeap<dist_t> &topCandidates,
        const size_t M
    ) {
        if (topCandidates.size() < M) {
            return;
        }

        // transform candidates from maxHeap to minHeap
        MinHeap<dist_t> queueClosest;
        std::vector<ID<dist_t>> returnList;
        while (topCandidates.size() > 0) {
            queueClosest.push(topCandidates.top());
            topCandidates.pop();
        }

        while (queueClosest.size()) {
            if (returnList.size() >= M)
                break;
            ID<dist_t> curID = queueClosest.top();
            dist_t curDist = curID.dist;
            queueClosest.pop();
            bool good = true;

            // for input new small node q, we have selected some candidates for its neighbors
            // for current candidate c, if there is a candidate o that is selected before and $dist(c, o) < dist(c, q)$"
            // then we skip this candidate c, otherwise select it
            size_t dim_ = dim;
            for (const ID<dist_t> otherID : returnList) {
                const void *curRawVec = static_cast<const void*>(rawVectors.get() + (curID.vecID * dim));
                const void *otherRawVec = static_cast<const void*>(rawVectors.get() + (otherID.vecID * dim));
                dist_t otherDist = distFunc(curRawVec, otherRawVec, &dim_);
                if (otherDist < curDist) {
                    good = false;
                    break;
                }
            }
            if (good) {
                returnList.push_back(curID);
            }
        }

        // int curM = M;
        // while (queueClosest.size() > 0 && curM) {
        //     ID<dist_t> curID = queueClosest.top();
        //     queueClosest.pop();
        //     returnList.push_back(curID);
        //     curM--;
        // }
        for (const ID<dist_t> curID : returnList) {
            topCandidates.push(curID);
        }
    }

    id_t mutuallyConnectNewElement(
        id_t curVecID,
        MaxHeap<dist_t> &topCandidates,
        const size_t curLevel
    ) {
        const size_t Mcurmax = (curLevel == 0) ? (size_t)numLevel0Edges : (size_t)batchSize;

        getNeighborsByHeuristic2(topCandidates, Mcurmax);
        if (topCandidates.size() > Mcurmax) {
            throw std::runtime_error("Heuristic returned more than Mcurmax neighbors");
        }

        std::vector<ID<dist_t>> selectedNeighbors;
        selectedNeighbors.reserve(Mcurmax);
        while (!topCandidates.empty()) {
            selectedNeighbors.push_back(topCandidates.top());
            topCandidates.pop();
        }

        if (selectedNeighbors.empty()) return curVecID;

        id_t next_closest_entry_point = selectedNeighbors.back().vecID;

        size_t dim_ = dim;

        /*
        --------------------------
        Level 0
        --------------------------
        */
        if (curLevel == 0) {

            auto &curEdge = nodes[curVecID].level0Edges;

            for (size_t i = 0; i < selectedNeighbors.size(); ++i) {
                curEdge.neighbors[i] = selectedNeighbors[i].vecID;
            }

            numNeighbors.at(curLevel).at(curVecID) = selectedNeighbors.size();

            for (size_t idx = 0; idx < selectedNeighbors.size(); ++idx) {

                const id_t otherVid = selectedNeighbors[idx].vecID;

                if (otherVid == curVecID) {
                    throw std::runtime_error("Trying to connect an element to itself");
                }

                std::unique_lock<std::mutex> lock(this->nodesLock[otherVid]);

                auto &otherEdge = nodes[otherVid].level0Edges;
                size_t &otherNumNeighbors = numNeighbors.at(curLevel).at(otherVid);

                if (otherNumNeighbors > Mcurmax) {
                    throw std::runtime_error("Bad value of numNeighbors");
                }

                if (otherNumNeighbors < Mcurmax) {

                    otherEdge.neighbors[otherNumNeighbors] = curVecID;
                    otherNumNeighbors++;

                } else {

                    const void *curRawVec   = rawVectors.get() + (curVecID * dim);
                    const void *otherRawVec = rawVectors.get() + (otherVid * dim);

                    dist_t curDist = distFunc(curRawVec, otherRawVec, &dim_);

                    MaxHeap<dist_t> candidates;
                    candidates.emplace(curVecID, curDist);

                    for (size_t j = 0; j < otherNumNeighbors; ++j) {

                        id_t otherNbr = otherEdge.neighbors[j];

                        dist_t d = distFunc(
                            rawVectors.get() + (otherVid  * dim),
                            rawVectors.get() + (otherNbr * dim),
                            &dim_
                        );

                        candidates.emplace(otherNbr, d);
                    }

                    getNeighborsByHeuristic2(candidates, Mcurmax);

                    size_t out = 0;
                    while (!candidates.empty()) {
                        otherEdge.neighbors[out] = candidates.top().vecID;
                        candidates.pop();
                        ++out;
                    }

                    otherNumNeighbors = out;
                }
            }
        }

        /*
        --------------------------
        Level > 0
        --------------------------
        */
        else {

            auto &curEdge = nodes[curVecID].edges[curLevel - 1];

            for (size_t i = 0; i < selectedNeighbors.size(); ++i) {
                curEdge.neighbors[i] = selectedNeighbors[i].vecID;
            }

            numNeighbors.at(curLevel).at(curVecID) = selectedNeighbors.size();

            for (size_t idx = 0; idx < selectedNeighbors.size(); ++idx) {

                const id_t otherVid = selectedNeighbors[idx].vecID;

                if (otherVid == curVecID) {
                    throw std::runtime_error("Trying to connect an element to itself");
                }

                std::unique_lock<std::mutex> lock(this->nodesLock[otherVid]);

                auto &otherEdge = nodes[otherVid].edges[curLevel - 1];
                size_t &otherNumNeighbors = numNeighbors.at(curLevel).at(otherVid);

                if (otherNumNeighbors > Mcurmax) {
                    throw std::runtime_error("Bad value of numNeighbors");
                }

                if (otherNumNeighbors < Mcurmax) {

                    otherEdge.neighbors[otherNumNeighbors] = curVecID;
                    otherNumNeighbors++;

                } else {

                    const void *curRawVec   = rawVectors.get() + (curVecID   * dim);
                    const void *otherRawVec = rawVectors.get() + (otherVid * dim);

                    dist_t curDist = distFunc(curRawVec, otherRawVec, &dim_);

                    MaxHeap<dist_t> candidates;
                    candidates.emplace(curVecID, curDist);

                    for (size_t j = 0; j < otherNumNeighbors; ++j) {

                        id_t otherNbr = otherEdge.neighbors[j];

                        dist_t d = distFunc(
                            rawVectors.get() + (otherVid * dim),
                            rawVectors.get() + (otherNbr * dim),
                            &dim_
                        );

                        candidates.emplace(otherNbr, d);
                    }

                    getNeighborsByHeuristic2(candidates, Mcurmax);

                    size_t out = 0;
                    while (!candidates.empty()) {
                        otherEdge.neighbors[out] = candidates.top().vecID;
                        candidates.pop();
                        ++out;
                    }

                    otherNumNeighbors = out;
                }
            }
        }

        return next_closest_entry_point;
    }

    struct NavigationAccuracyResult {
        int R = 1;
        double recallDistU8 = 0, recallDistU16 = 0;
        double avgTopRDistU8 = 0, avgTopRDistU16 = 0, avgTopRDistFP32 = 0; // r_th distance for each batch of neighbors, average across all nodes.
        double avgDistU8 = 0, avgDistU16 = 0, avgDistFP32 = 0; // distance from query to all nodes, average by each node

        static NavigationAccuracyResult avgByN(const std::vector<NavigationAccuracyResult> &results, const ssize_t n) {
            NavigationAccuracyResult finalRersult;
            for(auto result : results) {
                finalRersult.recallDistU8 += result.recallDistU8;
                finalRersult.recallDistU16 += result.recallDistU16;
                finalRersult.avgTopRDistU8 += result.avgTopRDistU8;
                finalRersult.avgTopRDistU16 += result.avgTopRDistU16;
                finalRersult.avgTopRDistFP32 += result.avgTopRDistFP32;
                finalRersult.avgDistU8 += result.avgDistU8;
                finalRersult.avgDistU16 += result.avgDistU16;
                finalRersult.avgDistFP32 += result.avgDistFP32;
            }
            finalRersult.recallDistU8 /= static_cast<double> (n);
            finalRersult.recallDistU16 /= static_cast<double> (n);
            finalRersult.avgTopRDistU8 /= static_cast<double> (n);
            finalRersult.avgTopRDistU16 /= static_cast<double> (n);
            finalRersult.avgTopRDistFP32 /= static_cast<double> (n);
            finalRersult.avgDistU8 /= static_cast<double> (n);
            finalRersult.avgDistU16 /= static_cast<double> (n);
            finalRersult.avgDistFP32 /= static_cast<double> (n);
            return finalRersult;
        }
    };
    NavigationAccuracyResult
    analyzeNavigationAccuracy(const void *query, const void *oQuery, const code_t *pqCodes, size_t R = 1) {
        assert(curElements > 0);


        computeLUT(static_cast<const dist_t*>(query), true);   // 8-bit LUT
        computeLUT(static_cast<const dist_t*>(query), false);  // 16-bit LUT

        if (R > batchSize) R = batchSize;

        size_t dim_ = dim;

        std::vector<dist_t>    distFP32(curElements);
        std::vector<qdist16_t> distU16 (curElements);
        std::vector<qdist_t>   distU8  (curElements);

        NavigationAccuracyResult result;
        result.avgTopRDistU8 = result.avgTopRDistU16 = result.avgTopRDistFP32 = 0;

        const int fullBatchEnd = (curElements / batchSize) * batchSize;
        #pragma omp parallel for schedule(static)
        for (int nodeID = 0; nodeID < fullBatchEnd; nodeID += batchSize) {
            code_t curCodes[batchSize*numSubspaces];
            std::copy_n(pqCodes + size_t(nodeID) * numSubspaces, batchSize*numSubspaces, curCodes);
            transpose(batchSize, numSubspaces, curCodes);

            alignas(64) qdist16_t d16[batchSize];
            search8BitAVX512_u16sum<numSubspaces, timing, batchSize>(q16LUTr, curCodes, d16);

            alignas(64) qdist_t d8[batchSize];
            search8BitAVX512<numSubspaces, true, batchSize>(q8LUTr, curCodes, d8);

            alignas(64) dist_t fp32[batchSize];

            for (int insideID = 0; insideID < batchSize; ++insideID) {
                const int idx = nodeID + insideID;
                const void* vec = static_cast<const void*>(rawVectors.get() + size_t(idx) * dim);

                distFP32[idx] = fp32[insideID] = distFunc(oQuery, vec, &dim_);
                distU16 [idx] = d16[insideID];
                distU8  [idx] = d8 [insideID];
            }
        }

        {
            const size_t fullBatchEnd = (size_t(curElements) / batchSize) * batchSize;
            const size_t tailCount    = size_t(curElements) - fullBatchEnd;

            if (tailCount > 0) {
                if (int(curElements) >= batchSize) {
                    const int startIdx = int(curElements) - batchSize; // 例如 1000-64=936
                    const uint8_t* codesPtr =
                        reinterpret_cast<const uint8_t*>(pqCodes) + size_t(startIdx) * numSubspaces;
                    code_t curCodes[batchSize*numSubspaces];
                    std::copy_n(codesPtr, batchSize*numSubspaces, curCodes);
                    transpose(batchSize, numSubspaces, curCodes);

                    alignas(64) qdist16_t d16[batchSize];
                    search8BitAVX512_u16sum<numSubspaces, timing, batchSize>(q16LUTr, curCodes, d16);

                    alignas(64) qdist_t d8[batchSize];
                    search8BitAVX512<numSubspaces, true, batchSize>(q8LUTr, curCodes, d8);

                    const int offset = batchSize - tailCount;  // 64 - 40 = 24
                    for (size_t t = 0; t < tailCount; ++t) {
                        const int idx = fullBatchEnd + t;      // 960..999
                        const void* vec = static_cast<const void*>(
                            rawVectors.get() + size_t(idx) * dim
                        );
                        distFP32[idx] = distFunc(oQuery, vec, &dim_);
                        distU16 [idx] = d16[offset + t];
                        distU8  [idx] = d8 [offset + t];
                    }
                } else {
                    alignas(64) uint8_t codesBuf[batchSize * numSubspaces];
                    std::memset(codesBuf, 0, sizeof(codesBuf));
                    const uint8_t* src = reinterpret_cast<const uint8_t*>(pqCodes);
                    std::memcpy(codesBuf, src, size_t(tailCount) * numSubspaces);

                    alignas(64) qdist16_t d16[batchSize];
                    search8BitAVX512_u16sum<numSubspaces, timing, batchSize>(q16LUTr, codesBuf, d16);
                    alignas(64) qdist_t d8[batchSize];
                    search8BitAVX512<numSubspaces, true, batchSize>(q8LUTr, codesBuf, d8);

                    for (size_t t = 0; t < tailCount; ++t) {
                        const size_t idx = t;
                        const void* vec = static_cast<const void*>(
                            rawVectors.get() + size_t(idx) * dim
                        );
                        distFP32[idx] = distFunc(oQuery, vec, &dim_);
                        distU16 [idx] = d16[t];
                        distU8  [idx] = d8 [t];
                    }
                }
            }
        }

        double sum8 = 0.0, sum16 = 0.0;
        double acc8 = 0.0, acc16 = 0.0, accfp = 0.0;
        #pragma omp parallel for reduction(+:sum8,sum16,acc8,acc16,accfp) schedule(static)
        for (int nodeID = 0; nodeID < int(curElements); ++nodeID) {
            const auto&curEdge = nodes[nodeID].edges[0];

            std::array<std::pair<dist_t,int>, batchSize> gt{};
            std::array<std::pair<qdist16_t,int>, batchSize> s16{};
            std::array<std::pair<qdist_t,int>, batchSize> s8{};
            std::array<uint8_t, batchSize> isGT{}; // {} make all values 0

            for (int i = 0; i < int(batchSize); ++i) {
                const id_t nid = curEdge.neighbors[i];
                gt [i] = { distFP32[nid], i }; 
                s16[i] = { distU16[nid], i };
                s8 [i] = { distU8 [nid], i };
            }

            std::partial_sort(gt.begin(),  gt.begin()  + R, gt.end(),  [](auto &a, auto &b){ return a.first < b.first; });
            std::partial_sort(s16.begin(), s16.begin() + R, s16.end(), [](auto &a, auto &b){ return a.first < b.first; });
            std::partial_sort(s8.begin(),  s8.begin()  + R, s8.end(),  [](auto &a, auto &b){ return a.first < b.first; });

            acc8 += s8[R - 1].first;
            acc16 += s16[R - 1].first;
            accfp += gt[R - 1].first;

            for (size_t k = 0; k < R; ++k) isGT[ gt[k].second ] = 1;

            int inter16 = 0;
            for (size_t k = 0; k < R; ++k) inter16 += isGT[ s16[k].second ];
            sum16 += double(inter16) / double(R);

            int inter8 = 0;
            for (size_t k = 0; k < R; ++k) inter8 += isGT[ s8[k].second ];
            sum8 += double(inter8) / double(R);
        }

        const double denom = (curElements > 0) ? double(curElements) : 1.0;

        result.R = R;
        result.recallDistU8 = sum8 / denom;
        result.recallDistU16 = sum16 / denom;
        result.avgTopRDistU8   = acc8  / denom;
        result.avgTopRDistU16  = acc16 / denom;
        result.avgTopRDistFP32 = accfp / denom;

        result.avgDistU8 = result.avgDistU16 = result.avgDistFP32 = 0;
        for(auto dist : distU8) result.avgDistU8 += (static_cast<double>(dist));
        for(auto dist : distU16) result.avgDistU16 += (static_cast<double>(dist));
        for(auto dist : distFP32) result.avgDistFP32 += dist;
        result.avgDistU8 /= denom; result.avgDistU16 /= denom; result.avgDistFP32 /= denom;

        return result;
    }

    size_t addPoints(
        const dist_t *pqCentroids,
        const uint8_t *pqCodes,
        const size_t numVectors,
        const dist_t *rawVectors,
        const std::vector<id_t> toExternalID
    ) {
        this->nodesLock = std::vector<std::mutex>(numVectors);

        this->toExternalID = toExternalID;
        this->toExternalID.shrink_to_fit();
        std::copy_n(pqCentroids, numCentroids * dim, this->pqCentroids);

        this->setRawVectors(rawVectors);

        this->nodes.resize(numVectors);

        size_t maxNonzeroLevels = 0;
        #pragma omp parallel for reduction(max:maxNonzeroLevels)
        for (id_t curID = 0; curID < numVectors; ++curID) {
            if (curID % 100000 == 0 && omp_get_thread_num() == 0) {
                printf("\r%d/%d \t\tvectors got random levels",
                    int(curID * omp_get_num_threads()), int(numVectors));
                std::fflush(stdout);
            }

            // 假设 getRandomLevel() 返回“非0层数量” (0 表示只有 level0)
            // 如果你的 getRandomLevel() 返回的是“最高层level”，那它也等价于非0层数量（HNSWlib 里就是这个语义）
            size_t nonzero = getRandomLevel();
            nodes[curID] = Node(nonzero);
            maxNonzeroLevels = std::max(maxNonzeroLevels, nonzero);
        }

        // 全局真实最高层就是 maxNonzeroLevels
        maxLevel = maxNonzeroLevels;
        printf("\nAll nodes got random levels, with maxlevel=%d\n", int(maxLevel));

        // numNeighbors[level][id]，level 是真实 level（含0）
        numNeighbors.clear();
        numNeighbors.resize(maxLevel + 1);
        for (size_t l = 0; l <= maxLevel; ++l) {
            numNeighbors[l].assign(numVectors, 0);
        }

        puts("#Neighbors Init Finished!");

        this->addPoint(0, rawVectors, numVectors);
        puts("First Node Added!");

        long long done = 0;

        #pragma omp parallel
        {
            visitedArray.init(numVectors);

            #pragma omp for schedule(guided, 64)
            for (id_t curID = 1; curID < numVectors; ++curID) {
                this->addPoint(curID, rawVectors + (curID * dim), numVectors);

                long long v;
                #pragma omp atomic capture
                v = ++done;

                if (v % 100000 == 0) {
                    printf("\r%lld/%d \tvectors are added", v, (int)numVectors);
                    fflush(stdout);
                }
            }
        }

        puts("\nAll vectors are added into the graph");
        curElements = numVectors;

        idGenerator = std::make_unique<IDGenerator>(numVectors);

        if constexpr (!enableNumNeighborsSerialization) {
            edgeAdjustment(numVectors);
        }

        loadNeighborsCodes(pqCodes);

        assert(maxLevelElementsCt <= batchSize);

        return maxLevelElementsCt;
    }

    std::vector<std::vector<size_t>> getNumNeighbors() const {
        return this->numNeighbors;
    }


    void addPoint(
        const id_t cur_c,
        const dist_t *curVector,
        const size_t numVectors
    ) {
        static std::mutex tempLock;
        tempLock.lock();

        this->nodesLock[cur_c].lock();

        const size_t curLevelNonzero = nodes[cur_c].getLevels(); // edges.size()
        const size_t curNodeMaxLevel = curLevelNonzero;          // 真实最高层：0..curNodeMaxLevel
        const size_t oldMaxLevel = maxLevel;                     // 真实最高层（全局）

        const void* query = static_cast<const void*>(curVector);
        size_t dim_ = dim;

        if (cur_c == 0) {
            entryPoint = cur_c;
            maxLevel = curNodeMaxLevel;
            tempLock.unlock();
            this->nodesLock[cur_c].unlock();
            return;
        }

        std::unique_lock<std::mutex> levelLock(globalLock);
        if (curNodeMaxLevel <= oldMaxLevel) levelLock.unlock();
        this->nodesLock[cur_c].unlock();
        tempLock.unlock();

        id_t curVecID = entryPoint;

        // ---------- Greedy from top level down to curNodeMaxLevel+1 ----------
        if (curNodeMaxLevel < oldMaxLevel) {
            dist_t curDist = distFunc(
                query,
                static_cast<const void*>(rawVectors.get() + (curVecID * dim)),
                &dim_
            );

            for (int level = (int)oldMaxLevel; level > (int)curNodeMaxLevel; --level) {
                bool changed = true;
                while (changed) {
                    changed = false;
                    std::unique_lock<std::mutex> lock(this->nodesLock[curVecID]);

                    // level > 0 这里必然成立（因为 level 从 oldMaxLevel 往下，且 oldMaxLevel>curNodeMaxLevel>=0）
                    // 新结构：edges[level-1]
                    auto &curEdge = nodes.at(curVecID).edges.at((size_t)level - 1);
                    const size_t numN = numNeighbors.at((size_t)level).at(curVecID);

                    // numN 不能超过 batchSize（Edge<batchSize>）
                    assert(numN <= batchSize);

                    for (size_t i = 0; i < numN; ++i) {
                        const id_t nextVecID = curEdge.neighbors[i];

                        dist_t tempDist = distFunc(
                            query,
                            static_cast<const void*>(rawVectors.get() + (nextVecID * dim)),
                            &dim_
                        );

                        if (tempDist < curDist) {
                            curDist = tempDist;
                            curVecID = nextVecID;
                            changed = true;
                        }
                    }
                }
            }
        }

        // ---------- Connect on levels min(curNodeMaxLevel, oldMaxLevel) .. 0 ----------
        for (int level = (int)std::min(curNodeMaxLevel, oldMaxLevel); level >= 0; --level) {
            MaxHeap<dist_t> topCandidates =
                searchKNNForAdding(static_cast<const void*>(curVector), level, curVecID, cur_c);

            curVecID = mutuallyConnectNewElement(cur_c, topCandidates, (size_t)level);
        }

        if (curNodeMaxLevel > maxLevel) {
            entryPoint = cur_c;
            maxLevel = curNodeMaxLevel;
        }
    }

    void supplementLayeredKHopEdges(const size_t level,
                                    const size_t alpha,
                                    size_t maxHops = std::numeric_limits<size_t>::max())
    {
        assert(level < numNeighbors.size());
        assert(numNeighbors[level].size() == curElements);

        std::vector<std::vector<id_t>> adj(curElements);
        adj.reserve(curElements);
        for (size_t u = 0; u < curElements; ++u) {
            const size_t deg = std::min(batchSize, numNeighbors[level][u]);
            adj[u].reserve(deg);
            for (size_t i = 0; i < deg; ++i) {
                id_t v = nodes[u].edges[level].neighbors[i];
                if (static_cast<size_t>(v) < curElements)
                    adj[u].push_back(v);
            }
        }

        if (maxHops > curElements) maxHops = curElements;

        std::vector<uint8_t> seen(curElements);
        std::vector<int>      dist(curElements);
        std::vector<id_t>     q;      
        std::vector<id_t>     layer;  
        std::vector<id_t>     candidates;

        for (size_t u = 0; u < curElements; ++u) {
            size_t &deg_u = numNeighbors[level][u];
            if (deg_u >= batchSize) continue;          
            if (deg_u == 0)         continue;         

            std::fill(seen.begin(), seen.end(), 0);
            std::fill(dist.begin(), dist.end(), -1);

            seen[u] = 1;
            const auto &nbrs = adj[u];
            for (id_t v : nbrs) {
                size_t vi = static_cast<size_t>(v);
                if (vi < curElements) seen[vi] = 1;
            }

            q.clear();
            for (id_t v : nbrs) {
                size_t vi = static_cast<size_t>(v);
                if (vi < curElements) {
                    dist[vi] = 1;
                    q.push_back(v);
                }
            }

            int head = 0;
            size_t visited = 0;

            for (size_t hop = 2; hop <= maxHops && deg_u < batchSize; ++hop) {
                layer.clear();
                int tail = static_cast<int>(q.size());
                for (; head < tail; ++head) {
                    id_t v = q[head];
                    size_t vi = static_cast<size_t>(v);
                    if (vi >= curElements) continue;

                    for (id_t w : adj[vi]) {
                        size_t wi = static_cast<size_t>(w);
                        if (wi >= curElements) continue;
                        if (dist[wi] != -1) continue;        
                        dist[wi] = static_cast<int>(hop);
                        q.push_back(w);
                        layer.push_back(w);
                        if (++visited >= curElements) break;  
                    }
                    if (visited >= curElements) break;
                }

                if (layer.empty()) break; 

                candidates.clear();
                for (id_t w : layer) {
                    size_t wi = static_cast<size_t>(w);
                    if (wi >= curElements) continue;
                    if (seen[wi]) continue;
                    candidates.push_back(w);
                }

                if (candidates.empty()) continue;

                size_t can_take = std::min({alpha, candidates.size(), batchSize - deg_u});
                for (size_t k = 0; k < can_take; ++k) {
                    nodes[u].edges[level].neighbors[deg_u] = candidates[k];
                    ++deg_u;
                    seen[static_cast<size_t>(candidates[k])] = 1; 
                    if (deg_u >= batchSize) break;
                }
            }
        }
    }


    template<size_t upEdgeNum>
    void supplementTwoHopEdges_parallel(const size_t level) {
        assert(level < numNeighbors.size());
        assert(numNeighbors[level].size() == curElements);

        // 每层的最大容量：L0=numLevel0Edges；L>0=batchSize（因为 Edge<batchSize>）
        const size_t cap = (level == 0) ? (size_t)numLevel0Edges : (size_t)batchSize;
        const size_t target = std::min(cap, (size_t)upEdgeNum);

        std::vector<size_t> initialDeg(curElements);
        for (size_t u = 0; u < curElements; ++u) {
            initialDeg[u] = std::min(numNeighbors[level][u], target);
        }

        constexpr size_t HCAP = 512;
        static_assert((HCAP & (HCAP - 1)) == 0, "HCAP must be power of two");
        constexpr id_t EMPTY = std::numeric_limits<id_t>::max();

        auto hash32 = [](uint32_t x) -> uint32_t {
            x ^= x >> 16;
            x *= 0x7feb352dU;
            x ^= x >> 15;
            x *= 0x846ca68bU;
            x ^= x >> 16;
            return x;
        };

        // helper：按 level 返回 neighbors 指针（L0 vs L>0，且 L>0 用 level-1）
        auto nbrPtr = [&](size_t vid) -> id_t* {
            if (level == 0) {
                return nodes[vid].level0Edges.neighbors;
            } else {
                return nodes[vid].edges[level - 1].neighbors;
            }
        };

        #pragma omp parallel
        {
            std::array<id_t, HCAP> table;

            auto clear_table = [&]() { table.fill(EMPTY); };

            auto contains = [&](id_t key) -> bool {
                uint32_t h = hash32((uint32_t)key);
                size_t pos = (size_t)h & (HCAP - 1);
                for (;;) {
                    id_t cur = table[pos];
                    if (cur == EMPTY) return false;
                    if (cur == key)   return true;
                    pos = (pos + 1) & (HCAP - 1);
                }
            };

            auto insert = [&](id_t key) {
                uint32_t h = hash32((uint32_t)key);
                size_t pos = (size_t)h & (HCAP - 1);
                for (;;) {
                    id_t cur = table[pos];
                    if (cur == key) return;
                    if (cur == EMPTY) { table[pos] = key; return; }
                    pos = (pos + 1) & (HCAP - 1);
                }
            };

            #pragma omp for schedule(static)
            for (size_t u = 0; u < curElements; ++u) {
                if (omp_get_thread_num() == 0 && (u % 1000000ULL == 0)) {
                    printf("\rSupplement 2-hop (level=%zu): %zu / %zu", level, u, curElements);
                    std::fflush(stdout);
                }

                const size_t deg_u_init = initialDeg[u];
                if (deg_u_init == 0) continue;

                size_t deg_u_local = deg_u_init;
                if (deg_u_local >= target) {
                    numNeighbors[level][u] = target;
                    continue;
                }

                clear_table();
                insert((id_t)u);

                id_t* nbrs_u = nbrPtr(u);

                // 先把已有邻居放进哈希表
                for (size_t i = 0; i < deg_u_init; ++i) {
                    id_t v = nbrs_u[i];
                    if ((size_t)v >= curElements) continue;
                    insert(v);
                }

                // 2-hop 补边：u -> v -> w
                for (size_t i = 0; i < deg_u_init && deg_u_local < target; ++i) {
                    id_t v = nbrs_u[i];
                    size_t vi = (size_t)v;
                    if (vi >= curElements) continue;

                    const size_t deg_v = initialDeg[vi];
                    const id_t* nbrs_v = nbrPtr(vi);

                    for (size_t j = 0; j < deg_v && deg_u_local < target; ++j) {
                        id_t w = nbrs_v[j];
                        size_t wi = (size_t)w;
                        if (wi >= curElements) continue;

                        if (contains(w)) continue;

                        // 写入 u 的邻居表（保证不超过 cap/target）
                        nbrs_u[deg_u_local++] = w;
                        insert(w);
                    }
                }

                numNeighbors[level][u] = deg_u_local;
            }
        }

        puts("\nSupplementTwoHopEdges done.");
    }

    template<size_t upEdgeNum>
    void supplementRandomEdges(const size_t maxLevel) {
        std::vector<std::vector<id_t>> candidates;
        candidates.resize(maxLevel + 1);
        for(int curID = 0; curID < curElements; ++curID) {
            size_t levels = nodes[curID].edges.size() - 1;
            levels = std::min(maxLevel, levels);
            for(int curLevel = 0; curLevel <= levels; ++curLevel) {
                candidates[curLevel].push_back(curID);
            }
        }

        for(id_t curID = 0; curID < curElements; ++curID) {
            if(curID % 10000 == 0 && omp_get_num_threads() == 0) {
                printf("\r%d/%d \t\t nodes are supplemented with edges           ", int(curID), int(curElements));
            }
            for(int curLevel = 0; curLevel <= maxLevel; ++curLevel){
                if(numNeighbors[curLevel][curID] == 0 || numNeighbors[curLevel][curID] == upEdgeNum) {
                    continue;
                }
                assert(curLevel <= (nodes[curID].edges.size() - 1));
                auto &curEdge = nodes[curID].edges[curLevel];
                std::vector<id_t> existedNeighbors(curEdge.neighbors, curEdge.neighbors + numNeighbors[curLevel][curID]);
                existedNeighbors.push_back(curID);
                std::vector<id_t> newNeighbors;
                if(curLevel != 0){
                    newNeighbors = idGenerator->getIDsFromCandidates(
                        upEdgeNum - numNeighbors[curLevel][curID],
                        existedNeighbors,
                        candidates[curLevel]
                    );
                } else {
                    newNeighbors = idGenerator->getIDs(
                        upEdgeNum - numNeighbors[curLevel][curID],
                        existedNeighbors
                    );
                }
                std::copy_n(newNeighbors.data(), upEdgeNum - numNeighbors[curLevel][curID], curEdge.neighbors + numNeighbors[curLevel][curID]);
                numNeighbors[curLevel][curID] = upEdgeNum;
            }
        }
        puts("\nAll nodes are supplemented randomly");
    }

    template<size_t upEdgeNum>
    void supplementRandomEdges_parallel(const size_t maxLevel) {
        std::vector<std::vector<id_t>> candidates(maxLevel + 1);

        for (size_t curID = 0; curID < curElements; ++curID) {
            const size_t nonzeroLevels = nodes[curID].edges.size();      // edges[0] 对应 level=1
            const size_t nodeMaxLevel  = std::min(maxLevel, nonzeroLevels);
            for (size_t l = 0; l <= nodeMaxLevel; ++l) {
                candidates[l].push_back((id_t)curID);
            }
        }

        constexpr size_t HCAP = 512;
        static_assert((HCAP & (HCAP - 1)) == 0, "HCAP must be power of two");
        constexpr id_t EMPTY = std::numeric_limits<id_t>::max();

        auto hash32 = [](uint32_t x) -> uint32_t {
            x ^= x >> 16; x *= 0x7feb352dU;
            x ^= x >> 15; x *= 0x846ca68bU;
            x ^= x >> 16;
            return x;
        };

        #pragma omp parallel
        {
            uint64_t seed = (uint64_t)std::random_device{}();
            seed ^= (uint64_t)omp_get_thread_num() * 0x9e3779b97f4a7c15ULL;
            std::mt19937_64 rng(seed);

            std::array<id_t, HCAP> table;

            auto clear_table = [&]() { table.fill(EMPTY); };

            auto contains = [&](id_t key) -> bool {
                uint32_t h = hash32((uint32_t)key);
                size_t pos = (size_t)h & (HCAP - 1);
                for (;;) {
                    id_t cur = table[pos];
                    if (cur == EMPTY) return false;
                    if (cur == key)   return true;
                    pos = (pos + 1) & (HCAP - 1);
                }
            };

            auto insert = [&](id_t key) {
                uint32_t h = hash32((uint32_t)key);
                size_t pos = (size_t)h & (HCAP - 1);
                for (;;) {
                    id_t cur = table[pos];
                    if (cur == key) return;
                    if (cur == EMPTY) { table[pos] = key; return; }
                    pos = (pos + 1) & (HCAP - 1);
                }
            };

            #pragma omp for schedule(dynamic, 1024)
            for (size_t curID = 0; curID < curElements; ++curID) {
                if ((curID % 10000ULL == 0) && omp_get_thread_num() == 0) {
                    printf("\r%zu/%zu nodes supplemented ...", curID, curElements);
                    std::fflush(stdout);
                }

                const size_t nonzeroLevels = nodes[curID].edges.size();
                const size_t upto = std::min(maxLevel, nonzeroLevels); // curLevel in [0..upto]

                for (size_t curLevel = 0; curLevel <= upto; ++curLevel) {

                    const size_t targetDeg =
                        (curLevel == 0) ? (size_t)numLevel0Edges : (size_t)upEdgeNum;

                    size_t deg = numNeighbors[curLevel][curID];
                    if (deg == 0 || deg >= targetDeg) continue;

                    clear_table();
                    insert((id_t)curID);

                    // 同层候选池（可选）
                    const std::vector<id_t>* poolPtr = nullptr;
                    if (!candidates[curLevel].empty()) poolPtr = &candidates[curLevel];

                    if (curLevel == 0) {
                        // -------- L0: level0Edges --------
                        auto &curEdge = nodes[curID].level0Edges;

                        for (size_t i = 0; i < deg; ++i) {
                            id_t v = curEdge.neighbors[i];
                            if ((size_t)v < curElements) insert(v);
                        }

                        size_t need = targetDeg - deg;
                        while (need > 0) {
                            id_t cand;
                            if (poolPtr) {
                                const auto &pool = *poolPtr;
                                cand = pool[(size_t)(rng() % pool.size())];
                            } else {
                                cand = (id_t)(rng() % curElements);
                            }

                            if ((size_t)cand >= curElements) continue;
                            if (contains(cand)) continue;

                            curEdge.neighbors[deg++] = cand;
                            insert(cand);
                            --need;
                        }

                    } else {
                        // -------- L>0: edges[curLevel-1] --------
                        auto &curEdge = nodes[curID].edges[curLevel - 1];

                        for (size_t i = 0; i < deg; ++i) {
                            id_t v = curEdge.neighbors[i];
                            if ((size_t)v < curElements) insert(v);
                        }

                        size_t need = targetDeg - deg;
                        while (need > 0) {
                            id_t cand;
                            if (poolPtr) {
                                const auto &pool = *poolPtr;
                                cand = pool[(size_t)(rng() % pool.size())];
                            } else {
                                cand = (id_t)(rng() % curElements);
                            }

                            if ((size_t)cand >= curElements) continue;
                            if (contains(cand)) continue;

                            curEdge.neighbors[deg++] = cand;
                            insert(cand);
                            --need;
                        }
                    }

                    numNeighbors[curLevel][curID] = targetDeg;
                }
            }
        }

        puts("\nAll nodes are supplemented randomly (parallel).");
    }

    void edgeAdjustment(const size_t numVectors) {
        // Step 1, make each node in first level connect to every other if it dose not has more than batchSize nodes.
        maxLevelElementsCt = 0;
        std::vector<id_t> maxLevelIDs;
        std::vector<id_t> secondMaxLevelIDs;
        for(int curID = 0; curID < curElements; ++curID) {
            if(maxLevel == (nodes[curID].edges.size())) {
                ++maxLevelElementsCt;
                maxLevelIDs.push_back(curID);
            }
            if(maxLevel - 1 == (nodes[curID].edges.size())) {
                secondMaxLevelIDs.push_back(curID);
            }
        }

        if(maxLevelIDs.size() >= batchSize) {
            maxLevel++;
            maxLevelElementsCt = maxLevelIDs.size() / batchSize;

            printf("We have %d elements in the max level, so level up to %d. Now max level has %d elements\n",
                 int(maxLevelIDs.size()), int(maxLevel), int(maxLevelElementsCt));
            
            for(int i = 0; i < maxLevelElementsCt; ++i) {
                maxLevelIDs[i] = maxLevelIDs[i*batchSize];
            }
            maxLevelIDs.resize(maxLevelElementsCt);
            entryPoint = maxLevelIDs[0];
            
            for(id_t mexLevelID : maxLevelIDs) {
                nodes[mexLevelID].edges.push_back(Edge<batchSize>{});
            }
            
            // We do not supplement edge for the max level, so may not need the numNeighbors.
            numNeighbors.push_back({});
            numNeighbors.back().assign(numVectors, 0);
        } else if(secondMaxLevelIDs.size() < batchSize) {
            printf("MaxLevel-1 has less than batchsize (%d) elements, so level down to %d\n", int(batchSize), int(maxLevel) - 1);
            maxLevel--;
            numNeighbors.pop_back();
            // drop the top-level edge block of the OLD max-level nodes first;
            // (previously maxLevelIDs was overwritten before this loop, which
            // popped edges of the NEW top-level nodes instead and corrupted
            // the heap once dense-connect wrote through edges[maxLevel-1])
            for(id_t maxLevelID : maxLevelIDs) {
                nodes[maxLevelID].edges.pop_back();
            }
            // the old max-level nodes still live on the new max level,
            // so the new top level is secondMaxLevelIDs + old maxLevelIDs
            secondMaxLevelIDs.insert(secondMaxLevelIDs.end(),
                                     maxLevelIDs.begin(), maxLevelIDs.end());
            maxLevelIDs = secondMaxLevelIDs;
            maxLevelElementsCt = maxLevelIDs.size();
        }

        printf("We have %d elements in level %d\n", int(maxLevelElementsCt), int(maxLevel));
        if(maxLevelElementsCt <= batchSize && maxLevelElementsCt != 1) {

            for(int curID : maxLevelIDs) {
                auto &curEdge = nodes[curID].edges[maxLevel - 1];
                int edgeCt = 0;
                for(int otherID : maxLevelIDs) {
                    curEdge.neighbors[edgeCt] = otherID;
                    edgeCt++;
                }
                numNeighbors[maxLevel][curID] = maxLevelElementsCt;
                for(; edgeCt < batchSize; ++edgeCt) {
                    curEdge.neighbors[edgeCt] = curID;
                }
            }
            puts("Max level dense conntected");
        }

        auto printAvgNumNeighbors = [&]() {
            for(int level = 0; level <= maxLevel; ++level) {
                size_t totalNeighbors = 0;
                int validNum = 0;
                for(int curVecID = 0; curVecID < curElements; ++curVecID) {
                    assert(numNeighbors[level][curVecID] <= batchSize);
                    if(numNeighbors[level][curVecID] > batchSize){
                        printf("Fatal Error: %d level with %d vector contains %d neighbors\n", int(level), int(curVecID), int(numNeighbors[level][curVecID]));
                    }
                    if(numNeighbors[level][curVecID] != 0){
                        totalNeighbors += numNeighbors[level][curVecID];
                        validNum++;
                        if(level == maxLevel) {
                            printf("For node %d, neighbors are: ", int(curVecID));
                            for(int i = 0; i < numNeighbors[level][curVecID]; ++i) {
                                printf("%u ", nodes[curVecID].edges[2].neighbors[i]);
                            }
                            puts("");
                        }
                    }
                }
                float avgNumNeighbors = totalNeighbors / static_cast<float>(validNum);
                printf("For Level %d, We have %d elements, avg #Neighbors is %.2f\n", int(level), int(validNum), avgNumNeighbors);
            }
        };

        puts("Layer-1 Supplement Done");
        for(size_t level = 0; level <= (maxLevel - 1); ++level) {
            supplementTwoHopEdges_parallel<batchSize>(level);
            supplementRandomEdges_parallel<batchSize>(level);
        }

    }


    void loadNeighborsCodes(const uint8_t *pqCodes) {
        for (id_t curID = 0; curID < curElements; ++curID) {
            const size_t nodeMaxLevel = nodes[curID].edges.size(); // 真实最高层

            // 1) 先装 level 0
            {
                auto &curEdge = nodes[curID].level0Edges;
                const size_t deg = numNeighbors[0][curID];
                for (size_t nID = 0; nID < deg; ++nID) {
                    id_t nbr = curEdge.neighbors[nID];
                    std::copy_n(
                        pqCodes + nbr * numSubspaces,
                        numSubspaces,
                        curEdge.neighborsCode + nID * numSubspaces
                    );
                }
                // 如果后面 SIMD 假设固定长度，剩余部分最好补 0
                for (size_t nID = deg; nID < numLevel0Edges; ++nID) {
                    std::fill_n(curEdge.neighborsCode + nID * numSubspaces,
                                numSubspaces, 0);
                }
                transpose(numLevel0Edges, numSubspaces, curEdge.neighborsCode);
            }

            // 2) 再装 level 1..nodeMaxLevel
            for (size_t level = 1; level <= nodeMaxLevel; ++level) {
                auto &curEdge = nodes[curID].edges[level - 1];
                const size_t deg = numNeighbors[level][curID];

                for (size_t nID = 0; nID < deg; ++nID) {
                    id_t nbr = curEdge.neighbors[nID];
                    std::copy_n(
                        pqCodes + nbr * numSubspaces,
                        numSubspaces,
                        curEdge.neighborsCode + nID * numSubspaces
                    );
                }
                for (size_t nID = deg; nID < batchSize; ++nID) {
                    std::fill_n(curEdge.neighborsCode + nID * numSubspaces,
                                numSubspaces, 0);
                }
                transpose(batchSize, numSubspaces, curEdge.neighborsCode);
            }
        }
        puts("All nodes' neighbors codes are loaded");
    }
};


}