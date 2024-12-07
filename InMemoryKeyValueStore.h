#ifndef IN_MEMORY_KEY_VALUE_STORE_H
#define IN_MEMORY_KEY_VALUE_STORE_H

#include <unordered_map>
#include <shared_mutex>
#include <string>
#include <vector>
#include <optional>
#include <immintrin.h>

class InMemoryKeyValueStore {
private:
    std::unordered_map<std::string, std::string> store;
    mutable std::shared_mutex mutex;

public:
    // Insert or update a key-value pair
    void put(const std::string& key, const std::string& value);

    // Retrieve the value associated with a key
    std::optional<std::string> get(const std::string& key) const;

    // Delete a key-value pair
    void del(const std::string& key);

    // Retrieve keys that match a given prefix
    std::vector<std::string> getKeysWithPrefix(const std::string& prefix) const;

    // SIMD-optimized prefix matching (internal helper)
    static bool simdPrefixMatch(const std::string& str, const std::string& prefix);
};

#endif
