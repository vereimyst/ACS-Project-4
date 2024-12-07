#include "DictionaryEncoder.h"
#include <iostream>
#include <random>
#include <string>
#include <chrono>
#include <vector>
#include <fstream>
#include <thread>
#include <cassert>

// Generate random strings for testing
std::vector<std::string> generateTestData(size_t count, size_t length) {
    const char charset[] = "abcdefghijklmnopqrstuvwxyzABCDEFGHIJKLMNOPQRSTUVWXYZ";
    std::mt19937 generator(std::random_device{}());
    std::uniform_int_distribution<int> distribution(0, sizeof(charset) - 2);

    std::vector<std::string> result(count);
    for (auto& str : result) {
        str.resize(length);
        for (auto& ch : str) {
            ch = charset[distribution(generator)];
        }
    }
    return result;
}

// Log results to a CSV file
void logToCSV(const std::string& filename, const std::string& testType, int threads, double time, const std::string& extra = "") {
    std::ofstream file(filename, std::ios::app);
    if (file.is_open()) {
        file << testType << "," << threads << "," << time << "," << extra << "\n";
    } else {
        std::cerr << "Error opening file: " << filename << "\n";
    }
}

// Test encoding performance across different thread counts
void testEncodingPerformance(DictionaryEncoder& encoder, const std::vector<std::string>& dataset, const std::string& csvFile) {
    for (int threads : {1, 2, 4, 8}) {
    // for (int threads : {1, 2, 3, 4, 5, 6, 7, 8, 9, 10, 11, 12, 13, 14, 15, 16}) {
        encoder.clear();
        auto start = std::chrono::high_resolution_clock::now();
        encoder.encode(dataset, threads);
        auto end = std::chrono::high_resolution_clock::now();
        double time = std::chrono::duration<double>(end - start).count();

        std::cout << "Encoding with " << threads << " threads took " << time << " seconds.\n";
        logToCSV(csvFile, "EncodingPerformance", threads, time);
    }
}

// Test encoding with different operational concurrency
void testConcurrency(DictionaryEncoder& encoder, const std::vector<std::string>& dataset, const std::string& csvFile) {
    const size_t operationsPerUser = 5000; // Number of operations each user performs
    const std::string targetValue = dataset[dataset.size() / 2]; // Target for single-item search
    const std::string prefix = "a"; // Prefix for prefix scans

    for (int users : {1, 2, 4, 8}) {
        auto start = std::chrono::high_resolution_clock::now();

        std::vector<std::thread> userThreads;

        for (int i = 0; i < users; ++i) {
            userThreads.emplace_back([&, i]() {
                for (size_t j = 0; j < operationsPerUser; ++j) {
                    if (j % 4 == 0) {
                        encoder.queryValueNonSIMD(targetValue); // Non-SIMD single-item search
                    } else if (j % 4 == 1) {
                        encoder.queryValueSIMD(targetValue); // SIMD single-item search
                    } else if (j % 4 == 2) {
                        encoder.queryPrefixNonSIMD(prefix); // Non-SIMD prefix scan
                    } else {
                        encoder.queryPrefixSIMD(prefix); // SIMD prefix scan
                    }
                }
            });
        }

        for (auto& thread : userThreads) {
            thread.join();
        }

        auto end = std::chrono::high_resolution_clock::now();
        double time = std::chrono::duration<double>(end - start).count();

        std::cout << "Operational concurrency with " << users << " users took " << time << " seconds.\n";
        logToCSV(csvFile, "OperationalConcurrencyTest", users, time);
    }
}

// Simulate different read vs. write ratios
void testReadWriteRatio(DictionaryEncoder& encoder, const std::vector<std::string>& dataset, const std::string& csvFile) {
    const int totalOperations = 10000; // Total number of operations
    const std::string targetValue = dataset[dataset.size() / 2]; // Target for single-item search
    const std::string prefix = "a"; // Prefix for prefix scans

    for (int readPercentage : {100, 90, 80, 50, 20, 0}) {
        int readCount = (totalOperations * readPercentage) / 100;
        int writeCount = totalOperations - readCount;

        auto start = std::chrono::high_resolution_clock::now();

        std::thread reader([&]() {
            for (int i = 0; i < readCount; ++i) {
                if (i % 4 == 0) {
                    encoder.queryValueNonSIMD(targetValue); // Non-SIMD single-item search
                } else if (i % 4 == 1) {
                    encoder.queryValueSIMD(targetValue); // SIMD single-item search
                } else if (i % 4 == 2) {
                    encoder.queryPrefixNonSIMD(prefix); // Non-SIMD prefix scan
                } else {
                    encoder.queryPrefixSIMD(prefix); // SIMD prefix scan
                }
            }
        });

        std::thread writer([&]() {
            for (int i = 0; i < writeCount; ++i) {
                encoder.encode(dataset, 1); // Simulate single-threaded writes
            }
        });

        reader.join();
        writer.join();

        auto end = std::chrono::high_resolution_clock::now();
        double time = std::chrono::duration<double>(end - start).count();

        std::cout << "Read:Write Ratio (" << readPercentage << "% reads) took " << time << " seconds.\n";
        logToCSV(csvFile, "ReadWriteTest", 1, time, std::to_string(readPercentage) + "% reads");
    }
}

// Test different value sizes
void testValueSizes(DictionaryEncoder& encoder, size_t numEntries, const std::string& csvFile) {
    for (size_t valueSize : {8, 64, 256}) {
        auto dataset = generateTestData(numEntries, valueSize);

        encoder.clear();
        auto start = std::chrono::high_resolution_clock::now();
        encoder.encode(dataset, 4); // Use 4 threads for all value size tests
        auto end = std::chrono::high_resolution_clock::now();
        double time = std::chrono::duration<double>(end - start).count();

        std::cout << "Encoding value size " << valueSize << " bytes took " << time << " seconds.\n";
        logToCSV(csvFile, "ValueSizeTest", 4, time, std::to_string(valueSize) + " bytes");
    }
}

void testQueryComparison(DictionaryEncoder& encoder, const std::vector<std::string>& dataset, const std::string& csvFile) {
    const std::string targetValue = dataset[dataset.size() / 2]; // Target value for the search
    const std::string prefix = "a"; // Prefix for prefix scan tests

    // Test vanilla column scan
    auto start = std::chrono::high_resolution_clock::now();
    int tmp_int = encoder.vanillaQueryValue(dataset, targetValue);
    auto end = std::chrono::high_resolution_clock::now();
    double vanillaTime = std::chrono::duration<double>(end - start).count();
    std::cout << "Vanilla Querying \"" << targetValue << "\" took " << vanillaTime << " seconds.\n";
    assert(tmp_int == dataset.size() / 2);
    logToCSV(csvFile, "VanillaColumnScan", 1, vanillaTime);

    // Test dictionary-based non-SIMD single-item search
    start = std::chrono::high_resolution_clock::now();
    tmp_int = encoder.queryValueNonSIMD(targetValue);
    end = std::chrono::high_resolution_clock::now();
    double nonSIMDTime = std::chrono::duration<double>(end - start).count();
    std::cout << "Non-SIMD Querying \"" << targetValue << "\" took " << nonSIMDTime << " seconds.\n";
    assert(tmp_int == dataset.size() / 2);
    logToCSV(csvFile, "QuerySingleItem", 1, nonSIMDTime, "Non-SIMD");

    // Test dictionary-based SIMD single-item search
    start = std::chrono::high_resolution_clock::now();
    tmp_int = encoder.queryValueSIMD(targetValue);
    end = std::chrono::high_resolution_clock::now();
    double simdSingleItemTime = std::chrono::duration<double>(end - start).count();
    std::cout << "SIMD Querying \"" << targetValue << "\" took " << simdSingleItemTime << " seconds.\n";
    assert(tmp_int == dataset.size() / 2);
    logToCSV(csvFile, "QuerySingleItem", 1, simdSingleItemTime, "SIMD");

    // Test vanilla prefix scan
    start = std::chrono::high_resolution_clock::now();
    std::vector<int> tmp_vec = encoder.vanillaQueryPrefix(dataset, prefix);
    end = std::chrono::high_resolution_clock::now();
    double vanillaPrefixTime = std::chrono::duration<double>(end - start).count();
    std::cout << "Vanilla Querying prefix \"" << prefix << "\" took " << vanillaPrefixTime << " seconds.\n";
    int expected_len = tmp_vec.size();
    logToCSV(csvFile, "VanillaPrefixScan", 1, vanillaPrefixTime);

    // Test dictionary-based non-SIMD prefix scan
    start = std::chrono::high_resolution_clock::now();
    tmp_vec = encoder.queryPrefixNonSIMD(prefix);
    end = std::chrono::high_resolution_clock::now();
    double nonSIMDPrefixTime = std::chrono::duration<double>(end - start).count();
    std::cout << "Non-SIMD Querying prefix \"" << prefix << "\" took " << nonSIMDPrefixTime << " seconds.\n";
    assert(expected_len == tmp_vec.size());
    logToCSV(csvFile, "QueryPrefixScan", 1, nonSIMDPrefixTime, "Non-SIMD");

    // Test dictionary-based SIMD prefix scan
    start = std::chrono::high_resolution_clock::now();
    tmp_vec = encoder.queryPrefixSIMD(prefix);
    end = std::chrono::high_resolution_clock::now();
    double simdPrefixTime = std::chrono::duration<double>(end - start).count();
    std::cout << "SIMD Querying prefix \"" << prefix << "\" took " << simdPrefixTime << " seconds.\n";
    assert(expected_len == tmp_vec.size());
    logToCSV(csvFile, "QueryPrefixScan", 1, simdPrefixTime, "SIMD");
}

int main() {
    DictionaryEncoder encoder;
    const std::string csvFile = "performance_results.csv";

    // Write CSV header
    std::ofstream file(csvFile, std::ios::trunc);
    if (file.is_open()) {
        file << "TestType,Users/Threads,Time(s),Extra\n";
        file.close();
    }

    size_t numEntries = 100000;
    auto testData = generateTestData(numEntries, 8); // Key size fixed at 8B

    // 1. Test encoding performance with different thread counts
    testEncodingPerformance(encoder, testData, csvFile);

    // // 2. Test operational concurrency (multiple users)
    // testConcurrency(encoder, testData, csvFile);

    // // 3. Test read vs. write ratios
    // testReadWriteRatio(encoder, testData, csvFile);

    // // 4. Test value sizes
    // testValueSizes(encoder, numEntries, csvFile);

    // 5. Test querying
    testQueryComparison(encoder, testData, csvFile);

    return 0;
}
