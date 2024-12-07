#include "InMemoryKeyValueStore.h"
#include <iostream>
#include <thread>
#include <vector>
#include <random>
#include <chrono>

// Function to generate random strings of fixed size
std::string generateRandomString(size_t length) {
    const char charset[] = "abcdefghijklmnopqrstuvwxyzABCDEFGHIJKLMNOPQRSTUVWXYZ0123456789";
    std::string result;
    result.resize(length);

    std::random_device rd;
    std::mt19937 generator(rd());
    std::uniform_int_distribution<> distribution(0, sizeof(charset) - 2);

    for (size_t i = 0; i < length; ++i) {
        result[i] = charset[distribution(generator)];
    }

    return result;
}

// Benchmark with varying operational concurrency
void benchmarkConcurrency(InMemoryKeyValueStore& kvStore, const std::vector<std::pair<std::string, std::string>>& data, int numUsers) {
    std::vector<std::thread> threads;
    auto task = [&kvStore, &data](int startIdx, int endIdx) {
        for (int i = startIdx; i < endIdx; ++i) {
            kvStore.put(data[i].first, data[i].second);
        }
    };

    int chunkSize = data.size() / numUsers;
    auto start = std::chrono::high_resolution_clock::now();

    for (int i = 0; i < numUsers; ++i) {
        threads.emplace_back(task, i * chunkSize, (i == numUsers - 1) ? data.size() : (i + 1) * chunkSize);
    }

    for (auto& t : threads) {
        t.join();
    }

    auto end = std::chrono::high_resolution_clock::now();
    std::chrono::duration<double> elapsed = end - start;

    std::cout << "Operational concurrency (" << numUsers << " users) took: " << elapsed.count() << " seconds.\n";
}

// Benchmark varying read vs. write ratios
void benchmarkReadWriteRatio(InMemoryKeyValueStore& kvStore, const std::vector<std::pair<std::string, std::string>>& data, int readPercent) {
    size_t numReads = (data.size() * readPercent) / 100;
    size_t numWrites = data.size() - numReads;

    std::vector<std::thread> threads;
    auto readTask = [&kvStore](size_t numReads, size_t startIdx) {
        for (size_t i = 0; i < numReads; ++i) {
            kvStore.get("key-" + std::to_string(startIdx + i));
        }
    };

    auto writeTask = [&kvStore, &data](size_t numWrites, size_t startIdx) {
        for (size_t i = 0; i < numWrites; ++i) {
            kvStore.put(data[startIdx + i].first, data[startIdx + i].second);
        }
    };

    auto start = std::chrono::high_resolution_clock::now();

    threads.emplace_back(readTask, numReads, 0);
    threads.emplace_back(writeTask, numWrites, numReads);

    for (auto& t : threads) {
        t.join();
    }

    auto end = std::chrono::high_resolution_clock::now();
    std::chrono::duration<double> elapsed = end - start;

    std::cout << "Read:Write Ratio (" << readPercent << "% reads) took: " << elapsed.count() << " seconds.\n";
}

// Benchmark value size
void benchmarkValueSize(InMemoryKeyValueStore& kvStore, size_t valueSize, size_t numEntries) {
    std::vector<std::pair<std::string, std::string>> data;
    for (size_t i = 0; i < numEntries; ++i) {
        data.emplace_back("key-" + std::to_string(i), generateRandomString(valueSize));
    }

    auto start = std::chrono::high_resolution_clock::now();

    for (const auto& [key, value] : data) {
        kvStore.put(key, value);
    }

    auto end = std::chrono::high_resolution_clock::now();
    std::chrono::duration<double> elapsed = end - start;

    std::cout << "Value size (" << valueSize << " bytes) insertion took: " << elapsed.count() << " seconds.\n";
}

int main() {
    InMemoryKeyValueStore kvStore;

    // Test data generation
    size_t numEntries = 10000;
    std::vector<std::pair<std::string, std::string>> testData;
    for (size_t i = 0; i < numEntries; ++i) {
        testData.emplace_back("key-" + std::to_string(i), generateRandomString(8)); // Default value size 8B
    }

    // 1. Test operational concurrency
    for (int numUsers : {1, 2, 4, 8}) {
        benchmarkConcurrency(kvStore, testData, numUsers);
    }

    // 2. Test read vs. write ratios
    for (int readPercent : {100, 90, 80, 50, 20, 0}) {
        benchmarkReadWriteRatio(kvStore, testData, readPercent);
    }

    // 3. Test value sizes
    for (size_t valueSize : {8, 64, 256}) {
        benchmarkValueSize(kvStore, valueSize, numEntries);
    }

    return 0;
}
