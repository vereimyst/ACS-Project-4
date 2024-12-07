# ACS-Project-4

Code Files: 
- DictionaryEncoder.h - header file for In-Memory Key-Value Store data structure
- DictionaryEncoder.cpp - implementation for In-Memory Key-Value Store data structure
- main.cpp - testbench and main file
- testbench.o - executable file

Compile with:
```
g++ -std=c++17 -mavx2 -pthread main.cpp DictionaryEncoder.cpp -o testbench
```
Note: the default number of threads tested is 1-16. Adjust accordingly if your device does not support this many threads.

Output Files:
- dictionary.txt - each row is an entry in the structure, formatting is [key, value]
- encoded_column.txt - the encoding of the original data values, can be adjusted in DictionaryEncoder::encode()
- performance_results.csv - a compiled summary of all test results

Other:
- performance_results.xlsx - a converted version of performance_results.csv to make graphs