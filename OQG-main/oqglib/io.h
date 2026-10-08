#include <iostream>
#include <fstream>
#include <string>
#include <sstream>

#pragma once

#ifndef IO_H
#define IO_H

namespace gg {
    float* read_fvecs(const std::string& file_path, int num_vectors, int dim) {
        std::ifstream file(file_path, std::ios::binary);
        if (!file.is_open()) {
            std::cerr << "Failed to open file!" << std::endl;
            exit(0);
        }

        file.read(reinterpret_cast<char*>(&num_vectors), sizeof(int));
        file.read(reinterpret_cast<char*>(&dim), sizeof(int));

        float *data = new float[dim*num_vectors];

        file.read(reinterpret_cast<char*>(data), num_vectors * dim * sizeof(float));

        file.close();
        return data;
    }

    void write_fvecs(const std::string& file_path, float* data, int num_vectors, int dim) {
        std::ofstream file(file_path, std::ios::binary);
        if (!file.is_open()) {
            std::cerr << "Failed to open file!" << std::endl;
            exit(0);
        }

        file.write(reinterpret_cast<const char*>(&num_vectors), sizeof(int));
        file.write(reinterpret_cast<const char*>(&dim), sizeof(int));

        file.write(reinterpret_cast<const char*>(data), num_vectors * dim * sizeof(float));

        file.close();
    }

    template <typename T>
    void writeValue(std::ofstream& ofs, const T& value) {
        ofs.write(reinterpret_cast<const char*>(&value), sizeof(T));
    }

    template <typename T>
    void readValue(std::ifstream& ifs, T& value) {
        ifs.read(reinterpret_cast<char*>(&value), sizeof(T));
    }

    template <typename T>
    void writeArray(std::ofstream& ofs, const T* arr, size_t size) {
        ofs.write(reinterpret_cast<const char*>(arr), sizeof(T) * size);
    }

    template <typename T>
    void readArray(std::ifstream& ifs, T* arr, size_t size) {
        ifs.read(reinterpret_cast<char*>(arr), sizeof(T) * size);
    }

    template <typename T>
    void writeVector(std::ofstream& ofs, const std::vector<T>& vec) {
        size_t size = vec.size();
        ofs.write(reinterpret_cast<const char*>(&size), sizeof(size)); 
        ofs.write(reinterpret_cast<const char*>(vec.data()), sizeof(T) * size);
    }

    template <typename T>
    void readVector(std::ifstream& ifs, std::vector<T>& vec) {
        size_t size;
        ifs.read(reinterpret_cast<char*>(&size), sizeof(size));           
        vec.resize(size);
        ifs.read(reinterpret_cast<char*>(vec.data()), sizeof(T) * size);
    }
}

#endif // IO_H