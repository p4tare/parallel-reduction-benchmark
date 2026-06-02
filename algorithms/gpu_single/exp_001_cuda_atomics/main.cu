#include <iostream>
#include <cstdint>

// Fallback just in case the python orchestrator fails to pass the macro
#ifndef DATA_TYPE
#define DATA_TYPE float
#endif

int main(int argc, char** argv) {
    std::cout << "[C++] CUDA Algorithm executed successfully!" << std::endl;
    std::cout << "[C++] Injected DATA_TYPE size: " << sizeof(DATA_TYPE) << " bytes." << std::endl;
    return 0;
}