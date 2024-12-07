#ifndef DICTIONARY_ENCODER_H
#define DICTIONARY_ENCODER_H

#include <mutex>
#include <thread>
#include <shared_mutex>
#include <atomic>
#include <chrono>
#include <fstream>
#include <iostream>
#include <vector>
#include <string>
#include <cstring>
#include <unordered_set>
#include <unordered_map>
#include <immintrin.h> // SIMD intrinsics
#include <algorithm> // For std::find
#include <optional>

class DictionaryEncoder {
private:
    std::unordered_map<std::string, int> dictionary; // Maps strings to IDs
    std::vector<int> encodedColumn;                 // Encoded data column
    mutable std::shared_mutex dictMutex;            // Mutex for thread-safe dictionary updates
    std::atomic<int> nextId = 0;                    // Atomic counter for dictionary IDs

public:
    // Encoding
    void encode(const std::vector<std::string>& column, int numThreads);
    void writeEncodedColumn(const std::string& filename);
    void writeDictionary(const std::string& filename);

    // Decoding
    std::vector<std::string> decode() const;

    // Query with and without SIMD
    int vanillaQueryValue(const std::vector<std::string>& column, const std::string& value);
    int queryValueNonSIMD(const std::string& value) const; // Single-item search (non-SIMD)
    int queryValueSIMD(const std::string& value) const;
    std::vector<int> vanillaQueryPrefix(const std::vector<std::string>& column, const std::string& prefix);
    std::vector<int> queryPrefixNonSIMD(const std::string& prefix) const; // Prefix scan (non-SIMD)
    std::vector<int> queryPrefixSIMD(const std::string& prefix) const; // Prefix scan (SIMD)

    // Helper
    void Put(const std::string& key, int value);
    std::optional<int> Get(const std::string& key) const;
    bool Delete(const std::string& key);
    void clear(); // Resets the dictionary and encoded column
};

#endif
