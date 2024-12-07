#include "DictionaryEncoder.h"
#include <fstream>
#include <iostream>
#include <thread>
#include <chrono>
#include <cstring>
#include <unordered_set>
#include <immintrin.h> // SIMD intrinsics
#include <algorithm> // For std::find

// Encode data into dictionary format using multi-threading
void DictionaryEncoder::encode(const std::vector<std::string>& column, int numThreads) {
    size_t chunkSize = column.size() / numThreads;
    std::vector<std::thread> threads;

    // Thread-local dictionaries for parallel encoding
    std::vector<std::unordered_map<std::string, int>> localDictionaries(numThreads);
    std::vector<std::vector<int>> localEncodedColumns(numThreads);

    // Parallel encoding
    for (int i = 0; i < numThreads; ++i) {
        size_t startIdx = i * chunkSize;
        size_t endIdx = (i == numThreads - 1) ? column.size() : (i + 1) * chunkSize;

        threads.emplace_back([&, i, startIdx, endIdx]() {
            for (size_t j = startIdx; j < endIdx; ++j) {
                const std::string& value = column[j];
                auto& localDict = localDictionaries[i];

                if (localDict.find(value) == localDict.end()) {
                    localDict[value] = localDict.size();
                }
                localEncodedColumns[i].push_back(localDict[value]);
            }
        });
    }

    for (auto& thread : threads) {
        thread.join();
    }

    // Merge thread-local dictionaries into the global dictionary
    std::unordered_map<std::string, int> mergedDictionary;
    for (const auto& localDict : localDictionaries) {
        for (const auto& [key, value] : localDict) {
            if (mergedDictionary.find(key) == mergedDictionary.end()) {
                mergedDictionary[key] = nextId++;
            }
        }
    }

    // Final encoding pass
    encodedColumn.clear();
    for (size_t i = 0; i < numThreads; ++i) {
        for (size_t j = 0; j < localEncodedColumns[i].size(); ++j) {
            const std::string& value = column[j + i * chunkSize];
            encodedColumn.push_back(mergedDictionary[value]);
        }
    }

    dictionary = std::move(mergedDictionary);
}

// Write the encoded column to a file
void DictionaryEncoder::writeEncodedColumn(const std::string& filename) {
    std::ofstream file(filename);
    if (!file.is_open()) {
        std::cerr << "Error opening file: " << filename << "\n";
        return;
    }
    for (const auto& id : encodedColumn) {
        file << id << "\n";
    }
    file.close();
}

// Write the dictionary to a file
void DictionaryEncoder::writeDictionary(const std::string& filename) {
    std::ofstream file(filename);
    if (!file.is_open()) {
        std::cerr << "Error opening file: " << filename << "\n";
        return;
    }
    for (const auto& [key, value] : dictionary) {
        file << key << "," << value << "\n";
    }
    file.close();
}

// Decode the encoded column back into strings
std::vector<std::string> DictionaryEncoder::decode() const {
    std::vector<std::string> decoded;
    decoded.reserve(encodedColumn.size());
    for (const auto& id : encodedColumn) {
        for (const auto& [key, value] : dictionary) {
            if (value == id) {
                decoded.push_back(key);
                break;
            }
        }
    }
    return decoded;
}

int DictionaryEncoder::vanillaQueryValue(const std::vector<std::string>& column, const std::string& value) {
    std::shared_lock lock(dictMutex);
    size_t index = 0;
    // Perform a linear search in the raw data column
    for (const auto& data : column) {
        if (data == value) {
            return static_cast<int>(index); // Return the first index of the match
        }
        ++index;
    }
    return -1; // Return -1 if the value is not found
}

// Non SIMD Single Value Search
int DictionaryEncoder::queryValueNonSIMD(const std::string& value) const {
    std::shared_lock lock(dictMutex);
    auto it = dictionary.find(value);
    if (it == dictionary.end()) {
        return -1;
    }

    size_t i = 0;
    for(; i < encodedColumn.size(); i++) {
        if (encodedColumn[i] == it->second) {
            return i;
        }
    }
    
    return -1;
}

int DictionaryEncoder::queryValueSIMD(const std::string& value) const {
    std::shared_lock lock(dictMutex);

    // Perform a dictionary lookup for the value
    auto it = dictionary.find(value);
    if (it == dictionary.end()) {
        return -1; // Value not found in dictionary
    }

    int dictValue = it->second; // Get the encoded value
    size_t n = encodedColumn.size();

    // Prepare SIMD register for the dictionary value
    __m256i targetVec = _mm256_set1_epi32(dictValue); // Set all 32-bit lanes to the target value

    // Process the encoded column in chunks of 8 (for 32-bit integers)
    size_t i = 0;
    for (; i + 7 < n; i += 8) {
        // Load 8 elements from the encoded column
        __m256i columnVec = _mm256_loadu_si256(reinterpret_cast<const __m256i*>(&encodedColumn[i]));

        // Compare with the target value
        __m256i cmpResult = _mm256_cmpeq_epi32(columnVec, targetVec);

        // Extract matching indices
        int mask = _mm256_movemask_epi8(cmpResult); // Create a mask of matching elements
        if (mask != 0) {
            // Return the index of the first match
            for (int j = 0; j < 8; ++j) {
                if (mask & (1 << (j * 4))) {
                    return static_cast<int>(i + j);
                }
            }
        }
    }

    // Scalar loop for remaining elements
    for (; i < n; ++i) {
        if (encodedColumn[i] == dictValue) {
            return static_cast<int>(i); // Return the index of the first match
        }
    }

    return -1; // No match found
}

// Baseline vanilla prefix scan on raw data
std::vector<int> DictionaryEncoder::vanillaQueryPrefix(const std::vector<std::string>& column, const std::string& prefix) {
    std::vector<int> results;
    size_t index = 0;

    for (const auto& value : column) {
        if (value.compare(0, prefix.size(), prefix) == 0) { // Check if the prefix matches
            results.push_back(index);
        }
        ++index;
    }

    return results;
}

// Non SIMD Prefix Query
std::vector<int> DictionaryEncoder::queryPrefixNonSIMD(const std::string& prefix) const {
    std::vector<int> matchingIndices;

    // Early exit for empty prefix - return all indices
    if (prefix.empty()) {
        matchingIndices.reserve(encodedColumn.size());
        for (size_t i = 0; i < encodedColumn.size(); ++i) {
            matchingIndices.push_back(static_cast<int>(i));
        }
        return matchingIndices;
    }

    // Step 1: Precompute matching dictionary indices
    std::unordered_set<int> matchingCodes;
    for (const auto& [dictWord, dictIndex] : dictionary) {
        // Check if dictionary word starts with the prefix
        if (dictWord.size() >= prefix.size() &&
            dictWord.compare(0, prefix.size(), prefix) == 0) {
            matchingCodes.insert(dictIndex);
        }
    }

    // Early exit if no matches
    if (matchingCodes.empty()) {
        return matchingIndices;
    }

    // Step 2: Single pass over encoded column to collect matching indices
    matchingIndices.reserve(encodedColumn.size()); // Reserve space for potential matches
    for (size_t i = 0; i < encodedColumn.size(); ++i) {
        if (matchingCodes.count(encodedColumn[i])) {
            matchingIndices.push_back(static_cast<int>(i));
        }
    }

    return matchingIndices;
}

std::vector<int> DictionaryEncoder::queryPrefixSIMD(const std::string& prefix) const {
    std::vector<int> results;
    std::shared_lock lock(dictMutex);

    // Step 1: Collect matching dictionary values into an unordered_set
    std::unordered_set<int> matchingCodes;
    alignas(32) char paddedPrefix[32] = {0};
    std::memcpy(paddedPrefix, prefix.data(), prefix.size());

    __m256i prefixVec = _mm256_loadu_si256(reinterpret_cast<const __m256i*>(paddedPrefix));

    for (const auto& [key, value] : dictionary) {
        if (key.size() >= prefix.size()) {
            alignas(32) char paddedKey[32] = {0};
            std::memcpy(paddedKey, key.data(), std::min(key.size(), sizeof(paddedKey)));

            __m256i keyVec = _mm256_loadu_si256(reinterpret_cast<const __m256i*>(paddedKey));
            __m256i cmpResult = _mm256_cmpeq_epi8(prefixVec, keyVec);

            int mask = _mm256_movemask_epi8(cmpResult);
            if ((mask & ((1 << prefix.size()) - 1)) == ((1 << prefix.size()) - 1)) {
                matchingCodes.insert(value);
            }
        }
    }

    // Early truncation if no matching codes found
    if (matchingCodes.empty()) {
        return results;
    }

    // Step 2: SIMD-optimized scan of the encodedColumn
    size_t n = encodedColumn.size();
    alignas(32) __m256i targetVecs[matchingCodes.size()];
    size_t idx = 0;
    for (int code : matchingCodes) {
        targetVecs[idx++] = _mm256_set1_epi32(code);
    }

    for (size_t i = 0; i + 7 < n; i += 8) {
        __m256i columnVec = _mm256_loadu_si256(reinterpret_cast<const __m256i*>(&encodedColumn[i]));

        // Compare column values with all targetVecs
        __m256i cmpResult = _mm256_setzero_si256();
        for (size_t j = 0; j < matchingCodes.size(); ++j) {
            cmpResult = _mm256_or_si256(cmpResult, _mm256_cmpeq_epi32(columnVec, targetVecs[j]));
        }

        // Extract matching indices
        int mask = _mm256_movemask_epi8(cmpResult);
        if (mask != 0) {
            for (int j = 0; j < 8; ++j) {
                if (mask & (1 << (j * 4))) {
                    results.push_back(static_cast<int>(i + j));
                }
            }
        }
    }

    // Scalar fallback for remaining elements
    for (size_t i = n - n % 8; i < n; ++i) {
        if (matchingCodes.count(encodedColumn[i])) {
            results.push_back(static_cast<int>(i));
        }
    }

    return results;
}

// Clear dictionary and encoded column
void DictionaryEncoder::clear() {
    dictionary.clear();
    encodedColumn.clear();
    nextId = 0;
}
