#include <iostream>
#include <pybind11/functional.h>
#include <pybind11/pybind11.h>
#include <pybind11/numpy.h>
#include <pybind11/stl.h>
#include <thread>
#include <atomic>
#include <stdlib.h>
#include <assert.h>
#include <filesystem>
#include <cmath>
#include <immintrin.h>   // _mm_malloc/_mm_free
#include <cblas.h>
#include <stdexcept>
#include <cstddef>
//#include "oqg.h"
#include "mn_bindings.hpp"

namespace py = pybind11;
using namespace pybind11::literals;  // needed to bring in _a literal

/*
This file is developed based on https://github.com/nmslib/hnswlib/.
We provide extra support here for 8-bit PQ and 16-bit distance accumulation.
*/


// template<class Function>
// inline void ParallelFor(size_t start, size_t end, size_t numThreads, Function fn) {
//     if (numThreads <= 0) {
//         numThreads = std::thread::hardware_concurrency();
//     }

//     if (numThreads == 1) {
//         for (size_t id = start; id < end; id++) {
//             fn(id, 0);
//         }
//     } else {
//         std::vector<std::thread> threads;
//         std::atomic<size_t> current(start);

//         // keep track of exceptions in threads
//         // https://stackoverflow.com/a/32428427/1713196
//         std::exception_ptr lastException = nullptr;
//         std::mutex lastExceptMutex;

//         for (size_t threadId = 0; threadId < numThreads; ++threadId) {
//             threads.push_back(std::thread([&, threadId] {
//                 while (true) {
//                     size_t id = current.fetch_add(1);

//                     if (id >= end) {
//                         break;
//                     }

//                     try {
//                         fn(id, threadId);
//                     } catch (...) {
//                         std::unique_lock<std::mutex> lastExcepLock(lastExceptMutex);
//                         lastException = std::current_exception();
//                         /*
//                          * This will work even when current is the largest value that
//                          * size_t can fit, because fetch_add returns the previous value
//                          * before the increment (what will result in overflow
//                          * and produce 0 instead of current + 1).
//                          */
//                         current = end;
//                         break;
//                     }
//                 }
//             }));
//         }
//         for (auto &thread : threads) {
//             thread.join();
//         }
//         if (lastException) {
//             std::rethrow_exception(lastException);
//         }
//     }
// }


// inline void assert_true(bool expr, const std::string & msg) {
//     if (expr == false) throw std::runtime_error("Unpickle Error: " + msg);
//     return;
// }


// inline void get_input_array_shapes(const py::buffer_info& buffer, size_t* rows, size_t* features) {
//     if (buffer.ndim != 2 && buffer.ndim != 1) {
//         char msg[256];
//         snprintf(msg, sizeof(msg),
//             "Input vector data wrong shape. Number of dimensions %d. Data must be a 1D or 2D array.",
//             buffer.ndim);
//         throw std::runtime_error(msg);
//     }
//     if (buffer.ndim == 2) {
//         *rows = buffer.shape[0];
//         *features = buffer.shape[1];
//     } else {
//         *rows = 1;
//         *features = buffer.shape[0];
//     }
// }

// inline std::vector<size_t> get_input_ids_and_check_shapes(const py::object& ids_, size_t feature_rows) {
//     std::vector<size_t> ids;
//     if (!ids_.is_none()) {
//         py::array_t < size_t, py::array::c_style | py::array::forcecast > items(ids_);
//         auto ids_numpy = items.request();
//         // check shapes
//         if (!((ids_numpy.ndim == 1 && ids_numpy.shape[0] == feature_rows) ||
//               (ids_numpy.ndim == 0 && feature_rows == 1))) {
//             char msg[256];
//             snprintf(msg, sizeof(msg),
//                 "The input label shape %d does not match the input data vector shape %d",
//                 ids_numpy.ndim, feature_rows);
//             throw std::runtime_error(msg);
//         }
//         // extract data
//         if (ids_numpy.ndim == 1) {
//             std::vector<size_t> ids1(ids_numpy.shape[0]);
//             for (size_t i = 0; i < ids1.size(); i++) {
//                 ids1[i] = items.data()[i];
//             }
//             ids.swap(ids1);
//         } else if (ids_numpy.ndim == 0) {
//             ids.push_back(*items.data());
//         }
//     }

//     return ids;
// }

// inline std::filesystem::path canon_path(const py::object& any_path) {
//     py::object fspath = py::module::import("os").attr("fspath")(any_path);
//     std::string s = py::cast<std::string>(fspath);
//     return std::filesystem::path(s);
// }




class GGIndex {
public:
    using dist_t = float;
    using id_t   = unsigned int;
    constexpr static size_t numCentroids = 256;
    constexpr static size_t batchSize    = 64;

private:
    template<size_t NumSubspaces, size_t dim>
    struct Impl {
        gg::GlobalGraph<batchSize, numCentroids, NumSubspaces, dim> hnsw;

        Impl(size_t M, size_t efConstruction) : hnsw(efConstruction, M) {}
        Impl() : hnsw(1, 1) {} 


        void save(const std::filesystem::path& path) {
            std::ofstream ofs(path, std::ios::binary);
            if (!ofs.is_open()) {
                throw std::runtime_error("BatchIndex::save: failed to open file for writing: " + path.string());
            }
            try { hnsw.save(ofs); } catch (...) { ofs.close(); throw; }
            ofs.close();
            if (!ofs) {
                throw std::runtime_error("BatchIndex::save: write failed (stream error) for: " + path.string());
            }
        }

        void load(const std::filesystem::path& path) {
            std::ifstream ifs(path, std::ios::binary);
            if (!ifs.is_open()) {
                throw std::runtime_error("BatchIndex::load: failed to open file for reading: " + path.string());
            }
            try { hnsw.load(ifs); } catch (...) { ifs.close(); throw; }
            ifs.close();
            if (!ifs) {
                throw std::runtime_error("BatchIndex::load: read failed (stream error) for: " + path.string());
            }
        }

        void setRawVectors(const dist_t* ptr) { hnsw.setRawVectors(ptr); }

        size_t addPoints(py::object pqCentroids, py::object pqCodes, size_t numVectors,
                       py::object rawVectors, py::object toExternalID)
        {
            auto pqCentroids_c = py::array_t<dist_t,  py::array::c_style | py::array::forcecast>(pqCentroids);
            auto pqCodes_c     = py::array_t<uint8_t, py::array::c_style | py::array::forcecast>(pqCodes);
            auto rawVectors_c  = py::array_t<dist_t,  py::array::c_style | py::array::forcecast>(rawVectors);

            auto ids_i64 = py::array_t<long long, py::array::c_style | py::array::forcecast>(toExternalID);
            if (ids_i64.ndim() != 1)
                throw std::invalid_argument("toExternalID must be 1-D");

            std::vector<id_t> id_vec;
            id_vec.reserve(static_cast<size_t>(ids_i64.shape(0)));
            const long long* id_src = ids_i64.data();
            for (ssize_t i = 0; i < ids_i64.shape(0); ++i) {
                long long v = id_src[i];
                if (v < 0) throw std::invalid_argument("toExternalID contains negative values");
                id_vec.push_back(static_cast<id_t>(v));
            }

            return hnsw.addPoints(
                pqCentroids_c.data(),
                pqCodes_c.data(),
                numVectors,
                rawVectors_c.data(),
                id_vec
            );
        }



        static inline float* rotate(
            const float* oQuery,   // (numQuery, dim) row-major
            const float* rotT,     // (dim, dim) row-major, already transposed
            const ssize_t numQuery,
            float *rotatedQueries
        ) {
            if (!oQuery || !rotT) throw std::runtime_error("null pointer input.");
            if (numQuery <= 0 || dim <= 0) throw std::runtime_error("invalid shape.");

            cblas_sgemm(
                CblasColMajor,
                CblasNoTrans, CblasNoTrans,
                dim,                // M  (rows of C_col)  = dim
                numQuery,           // N  (cols of C_col)  = numQuery
                dim,                // K
                1.0f,
                rotT, dim,          // A_col = rotT interpreted as ColMajor => actually rotT^T
                oQuery, dim,        // B_col = oQuery interpreted as ColMajor => actually oQuery^T
                0.0f,
                rotatedQueries, dim // C_col written as ColMajor (dim x numQuery)
            );

            return rotatedQueries;
        }


        py::tuple searchKNNPQ(py::object rot, py::array oQueries, size_t efSearch, size_t k, size_t numRefine)
        {
            auto oQ = py::array_t<dist_t, py::array::c_style | py::array::forcecast>(oQueries);
            if (oQ.ndim() != 2) throw std::invalid_argument("queries must be a 2-D numpy array");

            py::array_t<dist_t, py::array::c_style | py::array::forcecast> R;
            if(!rot.is_none()) {
                R = py::array_t<dist_t, py::array::c_style | py::array::forcecast>(rot);
            }

            const ssize_t n = oQ.shape(0), d = oQ.shape(1);
            py::array_t<unsigned int> out({n, static_cast<ssize_t>(k)});
            auto O = out.mutable_unchecked<2>();
            const float* Q = reinterpret_cast<float*>(
                _mm_malloc(static_cast<size_t>(n) * static_cast<size_t>(dim) * sizeof(float), 64)
            );

            double latency = 0.0;
            auto t0 = std::chrono::steady_clock::now();
            if(!rot.is_none()) {
                rotate(oQ.data(), R.data(), n, const_cast<float*>(Q));
            } else {
                _mm_free(const_cast<float*>(Q));
                Q = oQ.data();
            }

            auto t2 = std::chrono::steady_clock::now();
            double rotLatency = std::chrono::duration<double>(t2 - t0).count();

            
            for (ssize_t i = 0; i < n; ++i) {
                const void* qi  = static_cast<const void*>(Q + i * d);
                const void* oqi = static_cast<const void*>(oQ.data() + i * d);
                auto ans = hnsw.searchKNNPQ(qi, oqi, efSearch, k, numRefine);
                for (size_t j = 0; j < k; ++j) {
                    unsigned int v = (j < ans.size()) ? ans[j].vecID : 0;
                    O(i, j) = v;
                }
            }
            auto t1 = std::chrono::steady_clock::now();
            latency = std::chrono::duration<double>(t1 - t0).count();
            return py::make_tuple(out, latency);
        }

        py::tuple searchKNNPQ16(py::object rot, py::array oQueries, size_t efSearch, size_t k, size_t numRefine)
        {
            auto oQ = py::array_t<dist_t, py::array::c_style | py::array::forcecast>(oQueries);
            if (oQ.ndim() != 2) throw std::invalid_argument("queries must be a 2-D numpy array");

            py::array_t<dist_t, py::array::c_style | py::array::forcecast> R;
            if(!rot.is_none()) {
                R = py::array_t<dist_t, py::array::c_style | py::array::forcecast>(rot);
            }

            const ssize_t n = oQ.shape(0), d = oQ.shape(1);
            py::array_t<unsigned int> out({n, static_cast<ssize_t>(k)});
            auto O = out.mutable_unchecked<2>();
            const float* Q = reinterpret_cast<float*>(
                _mm_malloc(static_cast<size_t>(n) * static_cast<size_t>(dim) * sizeof(float), 64)
            );

            double latency = 0.0;
            auto t0 = std::chrono::steady_clock::now();
            if(!rot.is_none()) {
                rotate(oQ.data(), R.data(), n, const_cast<float*>(Q));
            } else {
                _mm_free(const_cast<float*>(Q));
                Q = oQ.data();
            }

            auto t2 = std::chrono::steady_clock::now();
            double rotLatency = std::chrono::duration<double>(t2 - t0).count();

            
            for (ssize_t i = 0; i < n; ++i) {
                const void* qi  = static_cast<const void*>(Q + i * d);
                const void* oqi = static_cast<const void*>(oQ.data() + i * d);
                auto ans = hnsw.searchKNNPQ16(qi, oqi, efSearch, k, numRefine);
                for (size_t j = 0; j < k; ++j) {
                    unsigned int v = (j < ans.size()) ? ans[j].vecID : 0;
                    O(i, j) = v;
                }
            }
            auto t1 = std::chrono::steady_clock::now();
            latency = std::chrono::duration<double>(t1 - t0).count();

            if constexpr (gg::timing){
                puts("Warning: Enable timing will introduce much overhead");
                printf("Total Latency is %lf secs\n", latency);
                printf("Rotation Latency is %lf secs\n", rotLatency);
                printf("16UKernal Latency is %lf secs\n", gg::kernal16Latency / (1e9));
                printf("LUT Construction latency is %lf secs\n", hnsw.lutLatency / (1e9));
                printf("Exact LUT Construction latency is %lf secs\n", hnsw.lutConsLatency / (1e9));
            }

            return py::make_tuple(out, latency);
        }

        py::tuple searchKNN(py::array oQueries, size_t efSearch, size_t k)
        {
            auto oQ = py::array_t<dist_t, py::array::c_style | py::array::forcecast>(oQueries);
            if (oQ.ndim() != 2) throw std::invalid_argument("queries must be a 2-D numpy array");

            const ssize_t n = oQ.shape(0), d = oQ.shape(1);
            py::array_t<unsigned int> out({n, static_cast<ssize_t>(k)});
            auto O = out.mutable_unchecked<2>();

            double latency = 0.0;
            auto t0 = std::chrono::steady_clock::now();
            for (ssize_t i = 0; i < n; ++i) {
                const void* oqi = static_cast<const void*>(oQ.data() + i * d);
                auto ans = hnsw.searchKNN(oqi, efSearch, k);
                for (size_t j = 0; j < k; ++j) {
                    unsigned int v = (j < ans.size()) ? ans[j].vecID : 0;
                    O(i, j) = v;
                }
            }
            auto t1 = std::chrono::steady_clock::now();
            latency = std::chrono::duration<double>(t1 - t0).count();
            return py::make_tuple(out, latency);
        }
    };


    using AnyImpl = std::variant<
        Impl<8,128>, Impl<8,192>, Impl<8,304>, Impl<8,784>, Impl<12,108>, Impl<12,132>, Impl<12,516>, Impl<16,128>, Impl<16,256>, Impl<16,512>, Impl<16,768>, Impl<20,140>, Impl<20,200>, Impl<20,260>, Impl<20,780>, Impl<20,1040>, Impl<24,120>, Impl<24,168>, Impl<24,264>, Impl<24,432>, Impl<24,2064>, Impl<28,280>, Impl<28,392>, Impl<28,784>, Impl<32,256>, Impl<36,288>, Impl<36,396>, Impl<40,160>, Impl<40,320>, Impl<48,960>, Impl<56,280>, Impl<60,180>, Impl<68,272>, Impl<68,544>
    >;

    AnyImpl impl;

    static AnyImpl makeImpl(size_t dim, size_t numSubspaces, size_t M, size_t efC) {
        switch (numSubspaces) {
            case 8:
                if (dim == 128) return AnyImpl{std::in_place_type<Impl<8,128>>, M, efC};
                if (dim == 192) return AnyImpl{std::in_place_type<Impl<8,192>>, M, efC};
                if (dim == 304) return AnyImpl{std::in_place_type<Impl<8,304>>, M, efC};
                if (dim == 784) return AnyImpl{std::in_place_type<Impl<8,784>>, M, efC};
                break;
            case 12:
                if (dim == 108) return AnyImpl{std::in_place_type<Impl<12,108>>, M, efC};
                if (dim == 132) return AnyImpl{std::in_place_type<Impl<12,132>>, M, efC};
                if (dim == 516) return AnyImpl{std::in_place_type<Impl<12,516>>, M, efC};
                break;
            case 16:
                if (dim == 128) return AnyImpl{std::in_place_type<Impl<16,128>>, M, efC};
                if (dim == 256) return AnyImpl{std::in_place_type<Impl<16,256>>, M, efC};
                if (dim == 512) return AnyImpl{std::in_place_type<Impl<16,512>>, M, efC};
                if (dim == 768) return AnyImpl{std::in_place_type<Impl<16,768>>, M, efC};
                break;
            case 20:
                if (dim == 140) return AnyImpl{std::in_place_type<Impl<20,140>>, M, efC};
                if (dim == 200) return AnyImpl{std::in_place_type<Impl<20,200>>, M, efC};
                if (dim == 260) return AnyImpl{std::in_place_type<Impl<20,260>>, M, efC};
                if (dim == 780) return AnyImpl{std::in_place_type<Impl<20,780>>, M, efC};
                if (dim == 1040) return AnyImpl{std::in_place_type<Impl<20,1040>>, M, efC};
                break;
            case 24:
                if (dim == 120) return AnyImpl{std::in_place_type<Impl<24,120>>, M, efC};
                if (dim == 168) return AnyImpl{std::in_place_type<Impl<24,168>>, M, efC};
                if (dim == 264) return AnyImpl{std::in_place_type<Impl<24,264>>, M, efC};
                if (dim == 432) return AnyImpl{std::in_place_type<Impl<24,432>>, M, efC};
                if (dim == 2064) return AnyImpl{std::in_place_type<Impl<24,2064>>, M, efC};
                break;
            case 28:
                if (dim == 280) return AnyImpl{std::in_place_type<Impl<28,280>>, M, efC};
                if (dim == 392) return AnyImpl{std::in_place_type<Impl<28,392>>, M, efC};
                if (dim == 784) return AnyImpl{std::in_place_type<Impl<28,784>>, M, efC};
                break;
            case 32:
                if (dim == 256) return AnyImpl{std::in_place_type<Impl<32,256>>, M, efC};
                break;
            case 36:
                if (dim == 288) return AnyImpl{std::in_place_type<Impl<36,288>>, M, efC};
                if (dim == 396) return AnyImpl{std::in_place_type<Impl<36,396>>, M, efC};
                break;
            case 40:
                if (dim == 160) return AnyImpl{std::in_place_type<Impl<40,160>>, M, efC};
                if (dim == 320) return AnyImpl{std::in_place_type<Impl<40,320>>, M, efC};
                break;
            case 48:
                if (dim == 960) return AnyImpl{std::in_place_type<Impl<48,960>>, M, efC};
                break;
            case 56:
                if (dim == 280) return AnyImpl{std::in_place_type<Impl<56,280>>, M, efC};
                break;
            case 60:
                if (dim == 180) return AnyImpl{std::in_place_type<Impl<60,180>>, M, efC};
                break;
            case 68:
                if (dim == 272) return AnyImpl{std::in_place_type<Impl<68,272>>, M, efC};
                if (dim == 544) return AnyImpl{std::in_place_type<Impl<68,544>>, M, efC};
                break;
            default:
                break;
        }
        printf("Input (numSubspaces, dim) is (%lu,%lu)\n", numSubspaces, dim);
        throw std::invalid_argument("Unsupported (numSubspaces, dim) combination");
    }
    static AnyImpl makeImplForLoad(size_t dim, size_t numSubspaces) {
        switch (numSubspaces) {
            case 8:
                if (dim == 128) return AnyImpl{std::in_place_type<Impl<8,128>>};
                if (dim == 192) return AnyImpl{std::in_place_type<Impl<8,192>>};
                if (dim == 304) return AnyImpl{std::in_place_type<Impl<8,304>>};
                if (dim == 784) return AnyImpl{std::in_place_type<Impl<8,784>>};
                break;
            case 12:
                if (dim == 108) return AnyImpl{std::in_place_type<Impl<12,108>>};
                if (dim == 132) return AnyImpl{std::in_place_type<Impl<12,132>>};
                if (dim == 516) return AnyImpl{std::in_place_type<Impl<12,516>>};
                break;
            case 16:
                if (dim == 128) return AnyImpl{std::in_place_type<Impl<16,128>>};
                if (dim == 256) return AnyImpl{std::in_place_type<Impl<16,256>>};
                if (dim == 512) return AnyImpl{std::in_place_type<Impl<16,512>>};
                if (dim == 768) return AnyImpl{std::in_place_type<Impl<16,768>>};
                break;
            case 20:
                if (dim == 140) return AnyImpl{std::in_place_type<Impl<20,140>>};
                if (dim == 200) return AnyImpl{std::in_place_type<Impl<20,200>>};
                if (dim == 260) return AnyImpl{std::in_place_type<Impl<20,260>>};
                if (dim == 780) return AnyImpl{std::in_place_type<Impl<20,780>>};
                if (dim == 1040) return AnyImpl{std::in_place_type<Impl<20,1040>>};
                break;
            case 24:
                if (dim == 120) return AnyImpl{std::in_place_type<Impl<24,120>>};
                if (dim == 168) return AnyImpl{std::in_place_type<Impl<24,168>>};
                if (dim == 264) return AnyImpl{std::in_place_type<Impl<24,264>>};
                if (dim == 432) return AnyImpl{std::in_place_type<Impl<24,432>>};
                if (dim == 2064) return AnyImpl{std::in_place_type<Impl<24,2064>>};
                break;
            case 28:
                if (dim == 280) return AnyImpl{std::in_place_type<Impl<28,280>>};
                if (dim == 392) return AnyImpl{std::in_place_type<Impl<28,392>>};
                if (dim == 784) return AnyImpl{std::in_place_type<Impl<28,784>>};
                break;
            case 32:
                if (dim == 256) return AnyImpl{std::in_place_type<Impl<32,256>>};
                break;
            case 36:
                if (dim == 288) return AnyImpl{std::in_place_type<Impl<36,288>>};
                if (dim == 396) return AnyImpl{std::in_place_type<Impl<36,396>>};
                break;
            case 40:
                if (dim == 160) return AnyImpl{std::in_place_type<Impl<40,160>>};
                if (dim == 320) return AnyImpl{std::in_place_type<Impl<40,320>>};
                break;
            case 48:
                if (dim == 960) return AnyImpl{std::in_place_type<Impl<48,960>>};
                break;
            case 56:
                if (dim == 280) return AnyImpl{std::in_place_type<Impl<56,280>>};
                break;
            case 60:
                if (dim == 180) return AnyImpl{std::in_place_type<Impl<60,180>>};
                break;
            case 68:
                if (dim == 272) return AnyImpl{std::in_place_type<Impl<68,272>>};
                if (dim == 544) return AnyImpl{std::in_place_type<Impl<68,544>>};
                break;
            default:
                break;
        }
        printf("Input (numSubspaces, dim) is (%lu,%lu)\n", numSubspaces, dim);
        throw std::invalid_argument("Unsupported (numSubspaces, dim) combination");
    }


public:
    GGIndex(size_t M, size_t efConstruction, size_t numSubspaces, size_t dim)
        : impl(makeImpl(dim, numSubspaces, M, efConstruction)) {}

    GGIndex(py::object filepath, size_t numSubspaces, size_t dim)
        : impl(makeImplForLoad(dim, numSubspaces))
    {
        const auto path = canon_path(filepath);
        std::visit([&](auto& x){ x.load(path); }, impl);
    }

    GGIndex(py::object filepath, py::object rawVectors, size_t numSubspaces, size_t dim)
        : impl(makeImplForLoad(dim, numSubspaces))
    {
        const auto path = canon_path(filepath);
        std::visit([&](auto& x){ x.load(path); }, impl);

        if (rawVectors.is_none()) {
            std::visit([&](auto& x){ x.setRawVectors(nullptr); }, impl);
        } else {
            //auto rv = py::array_t<dist_t, py::array::c_style | py::array::forcecast>(rawVectors);

            py::array_t<dist_t> rv(rawVectors);
            if (!(rv.flags() & py::array::c_style)) {
                throw std::runtime_error("Input must be C-contiguous");
            }
            auto buf = rv.request();
            if (buf.format != py::format_descriptor<dist_t>::format()) {
                throw std::runtime_error("Wrong dtype");
            }
            dist_t* ptr = static_cast<dist_t*>(buf.ptr);

            // std::visit([&](auto& x){ x.setRawVectors(rv.data()); }, impl);
            std::visit([&](auto& x){ x.setRawVectors(ptr); }, impl);
        }
    }


    void save(py::object filepath) {
        const auto path = canon_path(filepath);
        std::visit([&](auto& x){ x.save(path); }, impl);
    }

    void load(py::object filepath) {
        const auto path = canon_path(filepath);
        std::visit([&](auto& x){ x.load(path); }, impl);
    }

    void load(py::object filepath, py::object rawVectors) {
        load(filepath);
        if (rawVectors.is_none()) {
            std::visit([&](auto& x){ x.setRawVectors(nullptr); }, impl);
        } else {
            auto rv = py::array_t<dist_t, py::array::c_style | py::array::forcecast>(rawVectors);
            std::visit([&](auto& x){ x.setRawVectors(rv.data()); }, impl);
        }
    }

    size_t addPoints(py::object pqCentroids, py::object pqCodes, size_t numVectors,
                   py::object rawVectors, py::object toExternalID)
    {
        return std::visit([&](auto& x){ return x.addPoints(pqCentroids, pqCodes, numVectors, rawVectors, toExternalID); }, impl);
    }

    py::object searchKNNPQ(py::object rot, py::array oQueries, size_t efSearch, size_t k, size_t numRefine) {
        return std::visit([&](auto& x){ return x.searchKNNPQ(rot, oQueries, efSearch, k, numRefine); }, impl);
    }

    py::object searchKNNPQ16(py::object queries, py::array oQueries, size_t efSearch, size_t k, size_t numRefine) {
        return std::visit([&](auto& x){ return x.searchKNNPQ16(queries, oQueries, efSearch, k, numRefine); }, impl);
    }


    py::object searchKNN(py::array oQueries, size_t efSearch, size_t k) {
        return std::visit([&](auto& x){ return x.searchKNN(oQueries, efSearch, k); }, impl);
    }
};


// register one MNGGIndexT<batchSize, numLevel0Edges> instantiation; all
// (D, d) configurations share the exact same Python-facing interface
template<class Cls>
static void registerMNGGIndex(py::module& m, const char* name) {
    py::class_<Cls>(m, name)
        .def_readonly_static("numCentroids",  &Cls::numCentroids)
        .def_readonly_static("batchSize",     &Cls::batchSize)
        .def_readonly_static("numLevel0Edges", &Cls::numLevel0Edges)

        .def(py::init<size_t, size_t, size_t, size_t>())
        .def(py::init<py::object, size_t, size_t>())
        .def(py::init<py::object, py::object, size_t, size_t>())

        .def("save", &Cls::save)
        .def("load", py::overload_cast<py::object>(&Cls::load))
        .def("load", py::overload_cast<py::object, py::object>(&Cls::load))
        .def("addPoints", &Cls::addPoints)
        .def("memoryBreakdown", &Cls::memoryBreakdown)
        .def("searchKNNPQ", &Cls::searchKNNPQ)
        .def("searchKNN", &Cls::searchKNN)
        .def("searchKNNPQ16", &Cls::searchKNNPQ16);
}


PYBIND11_PLUGIN(oqglib) {
        py::module m("oqglib");
    
        py::class_<GGIndex>(m, "GGIndex")
            .def_readonly_static("numCentroids",  &GGIndex::numCentroids)
            .def_readonly_static("batchSize",     &GGIndex::batchSize)

            .def(py::init<size_t, size_t, size_t, size_t>())
            .def(py::init<py::object, size_t, size_t>())
            .def(py::init<py::object, py::object, size_t, size_t>())

            .def("save", &GGIndex::save)
            .def("load", py::overload_cast<py::object>(&GGIndex::load))
            .def("load", py::overload_cast<py::object, py::object>(&GGIndex::load))
            .def("addPoints", &GGIndex::addPoints)
            .def("searchKNNPQ", &GGIndex::searchKNNPQ)
            .def("searchKNN", &GGIndex::searchKNN)
            .def("searchKNNPQ16", &GGIndex::searchKNNPQ16);

        // default CBCA configuration: D(batchSize)=64, d(numLevel0Edges)=16
        registerMNGGIndex<MNGGIndex>(m, "MNGGIndex");

        // traditional full-width level-0 edge list (numLevel0Edges = 64),
        // exposed for the memory breakdown comparison (reviewer O3)
        registerMNGGIndex<MNGGIndexE64>(m, "MNGGIndexE64");

        // CBCA d = D/4 sensitivity: (D=128, d=32) and (D=256, d=64);
        // only the sift/glove/gist combos are compiled for these classes
        registerMNGGIndex<MNGGIndexB128E32>(m, "MNGGIndexB128E32");
        registerMNGGIndex<MNGGIndexB256E64>(m, "MNGGIndexB256E64");

        return m.ptr();
}
