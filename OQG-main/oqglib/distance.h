#pragma once

/*
This file is developed based on https://github.com/nmslib/hnswlib/.
We provide extra support here for 8-bit PQ and 16-bit distance accumulation.
*/

#ifndef __AVX512F__
#error "This file requires AVX-512 support (__AVX512F__ not defined)"
#endif

#if !defined(USE_AVX512) || USE_AVX512 == 0
#error "USE_AVX512 must be enabled (USE_AVX512=1)"
#endif

#if defined(USE_AVX512)
#include <immintrin.h>
#endif

#if defined(__GNUC__)
#define PORTABLE_ALIGN32 __attribute__((aligned(32)))
#define PORTABLE_ALIGN64 __attribute__((aligned(64)))
#else
#define PORTABLE_ALIGN32 __declspec(align(32))
#define PORTABLE_ALIGN64 __declspec(align(64))
#endif

namespace gg {

template<typename MTYPE>
using DISTFUNC = MTYPE(*)(const void *, const void *, const void *);

static float
L2Sqr(const void *pVect1v, const void *pVect2v, const void *qty_ptr) {
    float *pVect1 = (float *) pVect1v;
    float *pVect2 = (float *) pVect2v;
    size_t qty = *((size_t *) qty_ptr);

    float res = 0;
    for (size_t i = 0; i < qty; i++) {
        float t = *pVect1 - *pVect2;
        pVect1++;
        pVect2++;
        res += t * t;
    }
    return (res);
}

#if defined(USE_AVX512)

// Favor using AVX512 if available.
static float
L2SqrSIMD16ExtAVX512(const void *pVect1v, const void *pVect2v, const void *qty_ptr) {
    float *pVect1 = (float *) pVect1v;
    float *pVect2 = (float *) pVect2v;
    size_t qty = *((size_t *) qty_ptr);
    float PORTABLE_ALIGN64 TmpRes[16];
    size_t qty16 = qty >> 4;

    const float *pEnd1 = pVect1 + (qty16 << 4);

    __m512 diff, v1, v2;
    __m512 sum = _mm512_set1_ps(0);

    while (pVect1 < pEnd1) {
        v1 = _mm512_loadu_ps(pVect1);
        pVect1 += 16;
        v2 = _mm512_loadu_ps(pVect2);
        pVect2 += 16;
        diff = _mm512_sub_ps(v1, v2);
        // sum = _mm512_fmadd_ps(diff, diff, sum);
        sum = _mm512_add_ps(sum, _mm512_mul_ps(diff, diff));
    }

    _mm512_store_ps(TmpRes, sum);
    float res = TmpRes[0] + TmpRes[1] + TmpRes[2] + TmpRes[3] + TmpRes[4] + TmpRes[5] + TmpRes[6] +
            TmpRes[7] + TmpRes[8] + TmpRes[9] + TmpRes[10] + TmpRes[11] + TmpRes[12] +
            TmpRes[13] + TmpRes[14] + TmpRes[15];

    return (res);
}
#endif

#if defined(USE_AVX)

// Favor using AVX if available.
static float
L2SqrSIMD16ExtAVX(const void *pVect1v, const void *pVect2v, const void *qty_ptr) {
    float *pVect1 = (float *) pVect1v;
    float *pVect2 = (float *) pVect2v;
    size_t qty = *((size_t *) qty_ptr);
    float PORTABLE_ALIGN32 TmpRes[8];
    size_t qty16 = qty >> 4;

    const float *pEnd1 = pVect1 + (qty16 << 4);

    __m256 diff, v1, v2;
    __m256 sum = _mm256_set1_ps(0);

    while (pVect1 < pEnd1) {
        v1 = _mm256_loadu_ps(pVect1);
        pVect1 += 8;
        v2 = _mm256_loadu_ps(pVect2);
        pVect2 += 8;
        diff = _mm256_sub_ps(v1, v2);
        sum = _mm256_add_ps(sum, _mm256_mul_ps(diff, diff));

        v1 = _mm256_loadu_ps(pVect1);
        pVect1 += 8;
        v2 = _mm256_loadu_ps(pVect2);
        pVect2 += 8;
        diff = _mm256_sub_ps(v1, v2);
        sum = _mm256_add_ps(sum, _mm256_mul_ps(diff, diff));
    }

    _mm256_store_ps(TmpRes, sum);
    return TmpRes[0] + TmpRes[1] + TmpRes[2] + TmpRes[3] + TmpRes[4] + TmpRes[5] + TmpRes[6] + TmpRes[7];
}

#endif




static DISTFUNC<float> L2SqrSIMD16Ext;

static float
DoNothing(const void *pVect1v, const void *pVect2v, const void *qty_ptr) {return 0;}

class DoNothingSpace {
    size_t data_size_;
    size_t dim_;
public:
    DoNothingSpace(size_t dim) {
        dim_ = dim;
        data_size_ = dim * sizeof(float);
    }
    size_t get_data_size() {
        return data_size_;
    }
    DISTFUNC<float> get_dist_func() {
        return DoNothing;
    }
    void *get_dist_func_param() {
        return &dim_;
    }

    ~DoNothingSpace() {}
};

template<typename DType>
void printMatrixx(int row, int col, DType *data) {
    DType *curData = data;
    for(int i =0;i<row;++i){
        for(int j = 0; j<col;++j) {
            std::cout << *curData + 0 << " ";
            curData++;
        }
        std::cout << std::endl;
    }
}


extern volatile double kernal16Latency = 0;
template<int numSubspaces, bool timing, int batchSize = 64>
__mmask64 search8BitAVX512_u16sum(const __m512i* L, const uint8_t* codes, uint16_t* dists) {
    static_assert(batchSize % 64 == 0, "batchSize must be a multiple of 64");


    using clock = std::chrono::steady_clock;
    std::chrono::time_point<clock> t0;
    if constexpr(timing) {
        t0 = clock::now();
    }

    // codes 为 subspace-major 布局，步长为 batchSize；每次处理 64 个 code（一个 __m512i）
    for (int blk = 0; blk < batchSize / 64; ++blk) {
        // 两个 32-lane 的 u16 累加器
        __m512i acc_lo16 = _mm512_setzero_si512();
        __m512i acc_hi16 = _mm512_setzero_si512();

        for (int col = 0; col < numSubspaces; ++col) {
            __m512i code = _mm512_loadu_si512((const void*)(codes + batchSize * col + blk * 64));

            __mmask64 sel_mask = _mm512_test_epi8_mask(code, _mm512_set1_epi8(0x80));
            // idx = code & 0x7F
            __m512i idx = _mm512_and_si512(code, _mm512_set1_epi8(0x7F));

            __m512i r01 = _mm512_permutex2var_epi8(L[col*4+0], idx, L[col*4+1]);
            __m512i r23 = _mm512_permutex2var_epi8(L[col*4+2], idx, L[col*4+3]);
            __m512i r   = _mm512_mask_blend_epi8(sel_mask, r01, r23);   // 64×u8

            __m256i r_lo8  = _mm512_castsi512_si256(r);
            __m256i r_hi8  = _mm512_extracti64x4_epi64(r, 1);
            __m512i r_lo16 = _mm512_cvtepu8_epi16(r_lo8);   // vpmovzxbw
            __m512i r_hi16 = _mm512_cvtepu8_epi16(r_hi8);   // vpmovzxbw
            acc_lo16 = _mm512_add_epi16(acc_lo16, r_lo16);  // vpaddw
            acc_hi16 = _mm512_add_epi16(acc_hi16, r_hi16);  // vpaddw
        }

        // 回写 64×u16
        _mm512_storeu_si512((void*)(dists + blk * 64 +  0), acc_lo16);
        _mm512_storeu_si512((void*)(dists + blk * 64 + 32), acc_hi16);
    }

    if constexpr(timing){
        auto t1 = clock::now();
        double ms = std::chrono::duration<double, std::nano>(t1 - t0).count();
        kernal16Latency += ms;
    }

    return 0xFFFFFFFFFFFFFFFFULL;
}

template<int numSubspaces, bool timing>
__mmask64 search8BitAVX512_u16sumBatch(const __m512i* L, const uint8_t *codes1, const uint8_t *codes2, uint16_t* dists) {
    constexpr int batchSize = 32;


    using clock = std::chrono::steady_clock;
    std::chrono::time_point<clock> t0;
    if constexpr(timing) {
        t0 = clock::now();
    }

    // 两个 32-lane 的 u16 累加器
    __m512i acc_lo16 = _mm512_setzero_si512();
    __m512i acc_hi16 = _mm512_setzero_si512();

    
    constexpr static __mmask64 mA = ((__mmask64)1<<batchSize) - 1;
    constexpr static __mmask64 mB = ~mA;

    for (int col = 0; col < numSubspaces; ++col) {
        __m512i code = _mm512_setzero_si512();
        code = _mm512_mask_loadu_epi8(code, mA, codes1 + batchSize * col); 
        code = _mm512_mask_loadu_epi8(code, mB, codes2 + batchSize * col); 

        __mmask64 sel_mask = _mm512_test_epi8_mask(code, _mm512_set1_epi8(0x80));
        // idx = code & 0x7F
        __m512i idx = _mm512_and_si512(code, _mm512_set1_epi8(0x7F));

        __m512i r01 = _mm512_permutex2var_epi8(L[col*4+0], idx, L[col*4+1]);
        __m512i r23 = _mm512_permutex2var_epi8(L[col*4+2], idx, L[col*4+3]);
        __m512i r   = _mm512_mask_blend_epi8(sel_mask, r01, r23);   // 64×u8

        __m256i r_lo8  = _mm512_castsi512_si256(r);
        __m256i r_hi8  = _mm512_extracti64x4_epi64(r, 1);
        __m512i r_lo16 = _mm512_cvtepu8_epi16(r_lo8);   // vpmovzxbw
        __m512i r_hi16 = _mm512_cvtepu8_epi16(r_hi8);   // vpmovzxbw
        acc_lo16 = _mm512_add_epi16(acc_lo16, r_lo16);  // vpaddw
        acc_hi16 = _mm512_add_epi16(acc_hi16, r_hi16);  // vpaddw
    }

    _mm512_storeu_si512((void*)(dists +  0), acc_lo16);
    _mm512_storeu_si512((void*)(dists + 32), acc_hi16);

    if constexpr(timing){
        auto t1 = clock::now();
        double ms = std::chrono::duration<double, std::nano>(t1 - t0).count();
        kernal16Latency += ms;
    }

    return 0xFFFFFFFFFFFFFFFFULL;
}

template<int numSubspaces, bool keepValue, int batchSize = 64>
__mmask64 search8BitAVX512(const __m512i* lut_registers, const uint8_t* codes, uint8_t *dists) {
    static_assert(batchSize % 64 == 0, "batchSize must be a multiple of 64");
    static_assert(keepValue || batchSize == 64, "lowerbound pruning only supports batchSize == 64");

    __m512i lowerbound;
    if constexpr(!keepValue){
        lowerbound = _mm512_set1_epi8(*dists);
    }

    __mmask64 mask = 0xFFFFFFFFFFFFFFFF;

    // codes are subspace-major with stride batchSize; process 64 codes per block
    for (int blk = 0; blk < batchSize / 64; ++blk) {
        __m512i acc = _mm512_setzero_si512();

        for (int col = 0; col < numSubspaces; ++col) {
            // Load 64 PQ codes
            __m512i code_vector = _mm512_loadu_si512(reinterpret_cast<const void*>(codes + batchSize * col + blk * 64));

            // bytewise AND with 0b10000000
            __mmask64 sel_mask = _mm512_test_epi8_mask(code_vector, _mm512_set1_epi8(0x80));

            // bytewise AND with 0b01111111
            // _mm512_set1_epi8(0x7F) will create 64 0x7F and concatenate them as a 512 bit data
            __m512i idx = _mm512_and_si512(code_vector, _mm512_set1_epi8(0x7F));

            // Look up result from two LUT pairs
            __m512i result_01 = _mm512_permutex2var_epi8(lut_registers[col * 4 + 0], idx, lut_registers[col * 4 + 1]);
            __m512i result_23 = _mm512_permutex2var_epi8(lut_registers[col * 4 + 2], idx, lut_registers[col * 4 + 3]);

            // Blend based on sel_mask
            __m512i result = _mm512_mask_blend_epi8(sel_mask, result_01, result_23);

            acc = _mm512_adds_epu8(acc, result);
        }

        if constexpr(keepValue) {
            _mm512_store_si512(reinterpret_cast<__m512i*>(dists + blk * 64), acc);
        } else {
            mask = _mm512_cmplt_epu8_mask(acc, lowerbound);
            if(mask != 0){
                _mm512_store_si512(reinterpret_cast<__m512i*>(dists), acc);
            }
        }
    }

    return mask;

}


template<int numSubspaces, bool keepValue>
__mmask64 search8BitAVX512(const uint8_t* lut, const uint8_t* codes, uint8_t *dists) {
    constexpr int bathSize = 64;

    __m512i lowerbound;
    if constexpr(!keepValue){
        lowerbound = _mm512_set1_epi8(*dists);
    }

    //Step 1: preload 4 sub-LUTs for each subspace (each has 64 uint8 entries)
    __m512i lut_registers[4 * numSubspaces];
    for (int col = 0; col < numSubspaces; ++col) {
        for (int j = 0; j < 4; ++j) {
            lut_registers[col * 4 + j] = _mm512_loadu_si512(reinterpret_cast<const void*>(
                lut + col * 256 + j * 64));
        }
    }

    __m512i acc = _mm512_setzero_si512();

    for (int col = 0; col < numSubspaces; ++col) {
        // Load 64 PQ codes
        __m512i code_vector = _mm512_loadu_si512(reinterpret_cast<const void*>(codes + bathSize * col));

        // bytewise AND with 0b10000000
        __mmask64 sel_mask = _mm512_test_epi8_mask(code_vector, _mm512_set1_epi8(0x80));

        // bytewise AND with 0b01111111
        // _mm512_set1_epi8(0x7F) will create 64 0x7F and concatenate them as a 512 bit data
        __m512i idx = _mm512_and_si512(code_vector, _mm512_set1_epi8(0x7F));

        // Look up result from two LUT pairs
        __m512i result_01 = _mm512_permutex2var_epi8(lut_registers[col * 4 + 0], idx, lut_registers[col * 4 + 1]);
        __m512i result_23 = _mm512_permutex2var_epi8(lut_registers[col * 4 + 2], idx, lut_registers[col * 4 + 3]);

        // Blend based on sel_mask
        __m512i result = _mm512_mask_blend_epi8(sel_mask, result_01, result_23);

        acc = _mm512_adds_epu8(acc, result);
    }

    __mmask64 mask;

    if constexpr(keepValue) {
        _mm512_store_si512(reinterpret_cast<__m512i*>(dists), acc);
        mask = 0xFFFFFFFFFFFFFFFF;
    } else {
        mask = _mm512_cmplt_epu8_mask(acc, lowerbound);
        if(mask != 0){
            _mm512_store_si512(reinterpret_cast<__m512i*>(dists), acc);
        }
    }

    return mask;

}

template<int numSubspaces, bool keepValue>
__mmask64 search6BitAVX512(const uint8_t* lut, const uint8_t* codes, uint8_t *dists) {
    constexpr int batchSize = 64;

    // static double seconds = 0;
    // std::chrono::duration<double> duration;
    // auto start = std::chrono::high_resolution_clock::now();

    __m512i lowerbound;
    if constexpr(!keepValue){
        lowerbound = _mm512_set1_epi8(*dists);
    }

    // Step 1: 
    __m512i lut_registers[numSubspaces];
    for (int m = 0; m < numSubspaces; ++m) {
        lut_registers[m] = _mm512_loadu_si512(reinterpret_cast<const void*>(lut + m * 64));
    }

    __m512i acc = _mm512_setzero_si512();

    for (int col = 0; col < numSubspaces; ++col) {
        __m512i code_vec = _mm512_loadu_si512(reinterpret_cast<const void*>(codes + batchSize * col));
        __m512i result = _mm512_permutex2var_epi8(lut_registers[col], code_vec, lut_registers[col]);

        acc = _mm512_adds_epu8(acc, result);
    }

    __mmask64 mask;

    if constexpr(keepValue) {
        _mm512_store_si512(reinterpret_cast<__m512i*>(dists), acc);
        mask = 0xFFFFFFFFFFFFFFFF;
    } else {
        mask = _mm512_cmplt_epu8_mask(acc, lowerbound);
        if(mask != 0){
            _mm512_store_si512(reinterpret_cast<__m512i*>(dists), acc);
        }
    }

    // auto end = std::chrono::high_resolution_clock::now();
    // duration = end - start;
    // seconds += duration.count();
    // std::cout << seconds << std::endl;

    return mask;
}

template<int numCentroids, int numSubspaces, bool keepValue>
inline __mmask64 searchAVX512(const uint8_t* lut, const uint8_t* codes, uint8_t *dists){
    if constexpr(numCentroids == 256) {
        return search8BitAVX512<numSubspaces, keepValue>(lut, codes, dists);
    } else if constexpr(numCentroids == 64) {
        return search6BitAVX512<numSubspaces, keepValue>(lut, codes, dists);
    }
    exit(0);
    return 0;
}

class L2Space {
    DISTFUNC<float> fstdistfunc_;
    size_t data_size_;
    size_t dim_;

 public:
    L2Space(size_t dim) {
        fstdistfunc_ = L2Sqr;
#if defined(USE_AVX) || defined(USE_AVX512)
    #if defined(USE_AVX512)
        L2SqrSIMD16Ext = L2SqrSIMD16ExtAVX512;
        // else if (AVXCapable())
        //     L2SqrSIMD16Ext = L2SqrSIMD16ExtAVX;
    #elif defined(USE_AVX)
        if (AVXCapable())
            L2SqrSIMD16Ext = L2SqrSIMD16ExtAVX;
    #endif
#endif
        dim_ = dim;
        data_size_ = dim * sizeof(float);
    }

    size_t get_data_size() {
        return data_size_;
    }

    DISTFUNC<float> get_dist_func() {
        return fstdistfunc_;
    }

    void *get_dist_func_param() {
        return &dim_;
    }

    ~L2Space() {}
};

static int
L2SqrI4x(const void *__restrict pVect1, const void *__restrict pVect2, const void *__restrict qty_ptr) {
    size_t qty = *((size_t *) qty_ptr);
    int res = 0;
    unsigned char *a = (unsigned char *) pVect1;
    unsigned char *b = (unsigned char *) pVect2;

    qty = qty >> 2;
    for (size_t i = 0; i < qty; i++) {
        res += ((*a) - (*b)) * ((*a) - (*b));
        a++;
        b++;
        res += ((*a) - (*b)) * ((*a) - (*b));
        a++;
        b++;
        res += ((*a) - (*b)) * ((*a) - (*b));
        a++;
        b++;
        res += ((*a) - (*b)) * ((*a) - (*b));
        a++;
        b++;
    }
    return (res);
}

static int L2SqrI(const void* __restrict pVect1, const void* __restrict pVect2, const void* __restrict qty_ptr) {
    size_t qty = *((size_t*)qty_ptr);
    int res = 0;
    unsigned char* a = (unsigned char*)pVect1;
    unsigned char* b = (unsigned char*)pVect2;

    for (size_t i = 0; i < qty; i++) {
        res += ((*a) - (*b)) * ((*a) - (*b));
        a++;
        b++;
    }
    return (res);
}

class L2SpaceI {
    DISTFUNC<int> fstdistfunc_;
    size_t data_size_;
    size_t dim_;

 public:
    L2SpaceI(size_t dim) {
        if (dim % 4 == 0) {
            fstdistfunc_ = L2SqrI4x;
        } else {
            fstdistfunc_ = L2SqrI;
        }
        dim_ = dim;
        data_size_ = dim * sizeof(unsigned char);
    }

    size_t get_data_size() {
        return data_size_;
    }

    DISTFUNC<int> get_dist_func() {
        return fstdistfunc_;
    }

    void *get_dist_func_param() {
        return &dim_;
    }

    ~L2SpaceI() {}
};
}  // namespace gg
