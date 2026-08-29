#pragma once

#include <cerrno>
#include <cstring>
#include <stdexcept>
#include <string>
#include <vector>

#if defined(__linux__)
#include <pthread.h>
#include <sched.h>
#endif

namespace prbench {

inline std::vector<int> current_thread_affinity() {
#if defined(__linux__)
    cpu_set_t set;
    CPU_ZERO(&set);
    const int rc = pthread_getaffinity_np(pthread_self(), sizeof(set), &set);
    if (rc != 0) {
        throw std::runtime_error("pthread_getaffinity_np failed: " + std::string(std::strerror(rc)));
    }
    std::vector<int> cpus;
    for (int cpu = 0; cpu < CPU_SETSIZE; ++cpu) if (CPU_ISSET(cpu, &set)) cpus.push_back(cpu);
    return cpus;
#else
    return {};
#endif
}

class ScopedThreadAffinity {
public:
    explicit ScopedThreadAffinity(int cpu) : previous_(current_thread_affinity()) {
        if (cpu >= 0) pin(cpu);
    }
    explicit ScopedThreadAffinity(const std::vector<int>& cpus) : previous_(current_thread_affinity()) {
        if (!cpus.empty()) pin(cpus);
    }
    ~ScopedThreadAffinity() noexcept {
        try {
            if (!previous_.empty()) pin(previous_);
        } catch (...) {
        }
    }
    ScopedThreadAffinity(const ScopedThreadAffinity&) = delete;
    ScopedThreadAffinity& operator=(const ScopedThreadAffinity&) = delete;

private:
    static void pin(const std::vector<int>& cpus) {
#if defined(__linux__)
        cpu_set_t set;
        CPU_ZERO(&set);
        for (int cpu : cpus) CPU_SET(cpu, &set);
        const int rc = pthread_setaffinity_np(pthread_self(), sizeof(set), &set);
        if (rc != 0) throw std::runtime_error("pthread_setaffinity_np failed: " + std::string(std::strerror(rc)));
#else
        (void)cpus;
#endif
    }
    static void pin(int cpu) { pin(std::vector<int>{cpu}); }
    std::vector<int> previous_;
};

inline void pin_current_thread(const std::vector<int>& cpus) {
#if defined(__linux__)
    if (cpus.empty()) return;
    cpu_set_t set;
    CPU_ZERO(&set);
    for (int cpu : cpus) CPU_SET(cpu, &set);
    const int rc = pthread_setaffinity_np(pthread_self(), sizeof(set), &set);
    if (rc != 0) {
        throw std::runtime_error("pthread_setaffinity_np failed: " + std::string(std::strerror(rc)));
    }
#else
    (void)cpus;
#endif
}

inline void pin_current_thread(int cpu) {
    if (cpu < 0) return;
    pin_current_thread(std::vector<int>{cpu});
}

}  // namespace prbench
