#include "InMemoryKeyValueStore.h"
#include <iostream>
#include <cstring>
#include <chrono>

// Insert or update a key-value pair
void InMemoryKeyValueStore::put(const std::string& key, const std::string& value) {
    std::unique_lock lock(mutex);
    store[key] = value;
}

// Retrieve the value associated with a key
std::optional<std::string> InMemoryKeyValueStore::get(const std::string& key) const {
    std::shared_lock lock(mutex);
    auto it = store.find(key);
    if (it != store.end()) {
        return it->second;
    }
    return std::nullopt;
}

// Delete a key-value pair
void InMemoryKeyValueStore::del(const std::string& key) {
    std::unique_lock lock(mutex);
    store.erase(key);
}

// Retrieve keys that match a given prefix
std::vector<std::string> InMemoryKeyValueStore::getKeysWithPrefix(const std::string& prefix) const {
    std::vector<std::string> result;
    std::shared_lock lock(mutex);

    for (const auto& [key, _] : store) {
        if (simdPrefixMatch(key, prefix)) {
            result.push_back(key);
        }
    }
    return result;
}

// SIMD-optimized prefix matching
bool InMemoryKeyValueStore::simdPrefixMatch(const std::string& str, const std::string& prefix) {
    size_t prefixLen = prefix.length();
    if (str.length() < prefixLen) return false;

    const char* strData = str.data();
    const char* prefixData = prefix.data();

    for (size_t i = 0; i < prefixLen; i += 32) {
        __m256i strChunk = _mm256_loadu_si256(reinterpret_cast<const __m256i*>(strData + i));
        __m256i prefixChunk = _mm256_loadu_si256(reinterpret_cast<const __m256i*>(prefixData + i));

        __m256i result = _mm256_cmpeq_epi8(strChunk, prefixChunk);
        if (!_mm256_testc_si256(result, _mm256_set1_epi8(-1))) {
            return false;
        }
    }
    return true;
}
