#include <algorithm>
#include <chrono>
#include <cstdlib>
#include <iostream>
#include <sstream>
#include <stdexcept>
#include <string>
#include <utility>
#include <vector>

#if defined(__x86_64__) || defined(__i386__)
#if defined(__GNUC__) || defined(__clang__)
#include <cpuid.h>
#endif
#endif

#include "prbench/affinity.hpp"
#include "prbench/cli.hpp"
#include "prbench/cpu_backend.hpp"
#include "prbench/dataset.hpp"
#include "prbench/gpu_backend.hpp"
#include "prbench/metrics.hpp"
#include "prbench/strategy.hpp"

namespace {

using prbench::IterationMetrics;

std::string json_escape(const std::string& s) {
    std::ostringstream out;
    for (char c : s) {
        switch (c) {
            case '\\': out << "\\\\"; break;
            case '"': out << "\\\""; break;
            case '\n': out << "\\n"; break;
            case '\r': out << "\\r"; break;
            case '\t': out << "\\t"; break;
            default: out << c; break;
        }
    }
    return out.str();
}

void emit_prepare_metrics(const prbench::PrepareMetrics& p) {
    std::cout << "\"prepare_metrics\":{";
    std::cout << "\"calibration_samples\":[";
    for (std::size_t i = 0; i < p.calibration_samples.size(); ++i) {
        const auto& x = p.calibration_samples[i];
        std::cout << "{\"worker_kind\":\"" << json_escape(x.worker_kind)
                  << "\",\"worker_index\":" << x.worker_index
                  << ",\"device_id\":" << x.device_id
                  << ",\"elements\":" << x.elements
                  << ",\"elapsed_us\":" << x.elapsed_us
                  << ",\"throughput_elements_s\":" << x.throughput_elements_s << "}";
        if (i + 1 != p.calibration_samples.size()) std::cout << ',';
    }
    std::cout << "],\"linear_models\":[";
    for (std::size_t i = 0; i < p.linear_models.size(); ++i) {
        const auto& x = p.linear_models[i];
        std::cout << "{\"worker_kind\":\"" << json_escape(x.worker_kind)
                  << "\",\"worker_index\":" << x.worker_index
                  << ",\"device_id\":" << x.device_id
                  << ",\"intercept_us\":" << x.intercept_us
                  << ",\"slope_us_per_element\":" << x.slope_us_per_element << "}";
        if (i + 1 != p.linear_models.size()) std::cout << ',';
    }
    std::cout << "],\"partition\":[";
    for (std::size_t i = 0; i < p.partition.size(); ++i) {
        const auto& x = p.partition[i];
        std::cout << "{\"worker_kind\":\"" << json_escape(x.worker_kind)
                  << "\",\"worker_index\":" << x.worker_index
                  << ",\"device_id\":" << x.device_id
                  << ",\"offset\":" << x.offset
                  << ",\"elements\":" << x.elements << "}";
        if (i + 1 != p.partition.size()) std::cout << ',';
    }
    std::cout << "],\"initial_throughput_elements_s\":[";
    for (std::size_t i = 0; i < p.initial_throughput_elements_s.size(); ++i) {
        std::cout << p.initial_throughput_elements_s[i];
        if (i + 1 != p.initial_throughput_elements_s.size()) std::cout << ',';
    }
    std::cout << "]}";
}

void emit_result(std::size_t iteration, const IterationMetrics& m, const prbench::WorkerConfig& cfg) {
    double gpu_kernel_sum = 0.0;
    double gpu_h2d_sum = 0.0;
    double gpu_d2h_sum = 0.0;
    double gpu_total_max = 0.0;
    std::size_t gpu_elements_total = 0;
    for (const auto& d : m.gpus) {
        gpu_kernel_sum += d.kernel_us;
        gpu_h2d_sum += d.h2d_us;
        gpu_d2h_sum += d.d2h_us;
        gpu_total_max = std::max(gpu_total_max, d.total_us);
        gpu_elements_total += d.elements;
    }

    std::cout << "{\"event\":\"result\",\"iteration\":" << iteration
              << ",\"result\":" << m.result.json_literal()
              << ",\"e2e_us\":" << m.e2e_us
              << ",\"cpu_compute_us\":" << m.cpu.compute_us
              << ",\"cpu_chunks\":" << m.cpu.chunks
              << ",\"cpu_elements\":" << m.cpu.elements
              << ",\"gpu_elements_total\":" << gpu_elements_total
              << ",\"scheduler_us\":" << m.scheduler_us
              << ",\"merge_us\":" << m.merge_us
              << ",\"gpu_kernel_sum_us\":" << gpu_kernel_sum
              << ",\"gpu_h2d_sum_us\":" << gpu_h2d_sum
              << ",\"gpu_d2h_sum_us\":" << gpu_d2h_sum
              << ",\"gpu_total_max_us\":" << gpu_total_max
              << ",\"gpu_metrics\":[";
    for (std::size_t i = 0; i < m.gpus.size(); ++i) {
        const auto& d = m.gpus[i];
        std::cout << "{\"device_id\":" << d.device_id
                  << ",\"h2d_us\":" << d.h2d_us
                  << ",\"kernel_us\":" << d.kernel_us
                  << ",\"d2h_us\":" << d.d2h_us
                  << ",\"device_overhead_us\":" << d.device_overhead_us
                  << ",\"total_us\":" << d.total_us
                  << ",\"chunks\":" << d.chunks
                  << ",\"elements\":" << d.elements << "}";
        if (i + 1 != m.gpus.size()) std::cout << ',';
    }
    std::cout << "],\"worker_throughput_elements_s\":[";
    for (std::size_t i = 0; i < m.worker_throughput_elements_s.size(); ++i) {
        std::cout << m.worker_throughput_elements_s[i];
        if (i + 1 != m.worker_throughput_elements_s.size()) std::cout << ',';
    }
    std::cout << "],\"worker_throughput\":[";
    for (std::size_t i = 0; i < m.worker_throughput_elements_s.size(); ++i) {
        if (i == 0) {
            std::cout << "{\"worker_kind\":\"cpu\",\"worker_index\":0,\"device_id\":-1,\"throughput_elements_s\":"
                      << m.worker_throughput_elements_s[i] << "}";
        } else {
            const std::size_t gpu_index = i - 1;
            const int device_id = gpu_index < cfg.gpu_ids.size() ? cfg.gpu_ids[gpu_index] : -1;
            std::cout << "{\"worker_kind\":\"gpu\",\"worker_index\":" << gpu_index
                      << ",\"device_id\":" << device_id
                      << ",\"throughput_elements_s\":" << m.worker_throughput_elements_s[i] << "}";
        }
        if (i + 1 != m.worker_throughput_elements_s.size()) std::cout << ',';
    }
    std::cout << "]}" << std::endl;
}

std::pair<std::string, unsigned> detect_core_class_current_cpu() {
#if (defined(__x86_64__) || defined(__i386__)) && (defined(__GNUC__) || defined(__clang__))
    unsigned eax = 0, ebx = 0, ecx = 0, edx = 0;
    const unsigned max_basic = __get_cpuid_max(0, nullptr);
    if (max_basic < 7 || !__get_cpuid_count(7, 0, &eax, &ebx, &ecx, &edx)) {
        return {"homogeneous", 0};
    }
    const bool hybrid = (edx & (1u << 15u)) != 0u;
    if (!hybrid) return {"homogeneous", 0};
    if (max_basic < 0x1Au || !__get_cpuid_count(0x1A, 0, &eax, &ebx, &ecx, &edx)) {
        return {"unknown", 0};
    }
    const unsigned core_type = (eax >> 24u) & 0xffu;
    if (core_type == 0x20u) return {"efficiency", core_type};
    if (core_type == 0x40u) return {"performance", core_type};
    return {"unknown", core_type};
#else
    return {"unknown", 0};
#endif
}

int probe_cpu_types(const std::vector<int>& cpus) {
    std::cout << "{\"event\":\"cpu_types\",\"cpus\":[";
    for (std::size_t i = 0; i < cpus.size(); ++i) {
        prbench::pin_current_thread(cpus[i]);
        const auto [core_class, raw] = detect_core_class_current_cpu();
        std::cout << "{\"cpu_id\":" << cpus[i]
                  << ",\"core_class\":\"" << core_class
                  << "\",\"source\":\"native_cpuid\",\"raw_core_type\":" << raw << "}";
        if (i + 1 != cpus.size()) std::cout << ',';
    }
    std::cout << "]}" << std::endl;
    return 0;
}

int self_test() {
    std::vector<std::int32_t> ints{1, 2, 3, 4, 5};
    auto seq = prbench::reduce_cpu(
        ints.data(), ints.size(), prbench::DataType::Int32,
        prbench::CpuBackendKind::Sequential, 1, prbench::ReductionOperation::Sum
    );
    if (std::get<std::int64_t>(seq.result.storage) != 15) return 1;
    auto seq_min = prbench::reduce_cpu(
        ints.data(), ints.size(), prbench::DataType::Int32,
        prbench::CpuBackendKind::Sequential, 1, prbench::ReductionOperation::Min
    );
    if (std::get<std::int64_t>(seq_min.result.storage) != 1) return 2;
    auto seq_max = prbench::reduce_cpu(
        ints.data(), ints.size(), prbench::DataType::Int32,
        prbench::CpuBackendKind::Sequential, 1, prbench::ReductionOperation::Max
    );
    if (std::get<std::int64_t>(seq_max.result.storage) != 5) return 3;
#if PRBENCH_HAS_OPENMP
    auto omp = prbench::reduce_cpu(
        ints.data(), ints.size(), prbench::DataType::Int32,
        prbench::CpuBackendKind::OpenMPSimd, 2, prbench::ReductionOperation::Sum
    );
    if (std::get<std::int64_t>(omp.result.storage) != 15) return 4;
    auto omp_min = prbench::reduce_cpu(
        ints.data(), ints.size(), prbench::DataType::Int32,
        prbench::CpuBackendKind::OpenMPSimd, 2, prbench::ReductionOperation::Min
    );
    if (std::get<std::int64_t>(omp_min.result.storage) != 1) return 5;
    auto omp_max = prbench::reduce_cpu(
        ints.data(), ints.size(), prbench::DataType::Int32,
        prbench::CpuBackendKind::OpenMPSimd, 2, prbench::ReductionOperation::Max
    );
    if (std::get<std::int64_t>(omp_max.result.storage) != 5) return 6;
#endif
    // Regression coverage for the dynamic scheduler capacity bug discovered on
    // a real CUDA server: fixed/guided scheduling must not run the adaptive
    // throughput calibration, while adaptive calibration must never exceed its
    // configured maximum chunk size.
    if (prbench::dynamic_calibration_elements(
            prbench::SchedulerKind::DynamicFixed, 1'000'000, 65'536, 16'777'216) != 0) return 7;
    if (prbench::dynamic_calibration_elements(
            prbench::SchedulerKind::DynamicGuided, 1'000'000, 65'536, 16'777'216) != 0) return 8;
    if (prbench::dynamic_calibration_elements(
            prbench::SchedulerKind::DynamicAdaptive, 100'000'000, 65'536, 262'144) != 262'144) return 9;
    if (prbench::dynamic_calibration_elements(
            prbench::SchedulerKind::DynamicAdaptive, 100'000, 65'536, 16'777'216) != 100'000) return 10;
    std::cout << "{\"event\":\"self_test\",\"status\":\"ok\"}" << std::endl;
    return 0;
}

}  // namespace

int main(int argc, char** argv) {
    try {
        const auto cfg = prbench::parse_cli(argc, argv);
        if (cfg.self_test) return self_test();
        if (cfg.build_info) {
            std::cout << "{\"event\":\"build_info\",\"gpu_libraries\":"
                      << prbench::gpu_library_versions_json() << "}" << std::endl;
            return 0;
        }
        if (cfg.probe_cpu_types) return probe_cpu_types(cfg.probe_cpus);

        // The process-level affinity defines the CPU compute pool for OpenMP strategies.
        // A sequential CPU baseline is pinned to one processing unit to prevent migration
        // between cores. GPU-only runs pin the host control thread near the selected GPU
        // whenever the orchestrator supplied a locality-aware control CPU.
        if (cfg.scheduler == prbench::SchedulerKind::CpuOnly &&
            cfg.cpu_backend == prbench::CpuBackendKind::Sequential &&
            !cfg.cpu_affinity.empty()) {
            prbench::pin_current_thread(cfg.cpu_affinity.front());
        } else if (cfg.cpu_affinity.empty() && !cfg.gpu_worker_cpus.empty()) {
            // GPU-only and pure multi-GPU schedulers have no CPU compute pool.  Keep the
            // orchestration/merge thread on a deterministic topology-local control core.
            prbench::pin_current_thread(cfg.gpu_worker_cpus.front());
        } else if (!cfg.cpu_affinity.empty()) {
            prbench::pin_current_thread(cfg.cpu_affinity);
        }

        const bool needs_gpu = cfg.scheduler != prbench::SchedulerKind::CpuOnly;
        if (needs_gpu && !prbench::gpu_runtime_available()) {
            std::cout << "{\"event\":\"unsupported\",\"reason\":\""
                      << json_escape(prbench::gpu_unavailable_reason()) << "\"}" << std::endl;
            return 0;
        }

        prbench::Dataset dataset(cfg.dataset_path, cfg.count, cfg.dtype);
        const auto create_start = std::chrono::steady_clock::now();
        auto strategy = prbench::make_strategy(cfg, dataset);
        const auto create_end = std::chrono::steady_clock::now();
        const auto prepare_start = std::chrono::steady_clock::now();
        strategy->prepare(cfg.warmup_runs);
        const auto prepare_end = std::chrono::steady_clock::now();
        const double strategy_create_us =
            std::chrono::duration<double, std::micro>(create_end - create_start).count();
        const double prepare_us =
            std::chrono::duration<double, std::micro>(prepare_end - prepare_start).count();

        std::cout << "{\"event\":\"ready\",\"warmup_median_us\":"
                  << strategy->warmup_median_us()
                  << ",\"strategy_create_us\":" << strategy_create_us
                  << ",\"prepare_us\":" << prepare_us << ",";
        emit_prepare_metrics(strategy->prepare_metrics());
        std::cout << "}" << std::endl;

        std::string command;
        if (!std::getline(std::cin, command)) throw std::runtime_error("control pipe closed before PROBE/TIMING");
        std::string verb;
        {
            std::istringstream first_cmd(command);
            std::size_t probe_repetitions = 0;
            first_cmd >> verb >> probe_repetitions;
            if (verb == "PROBE") {
                if (probe_repetitions == 0) throw std::runtime_error("PROBE repetitions must be positive");
                const auto probe_start = std::chrono::steady_clock::now();
                double sum_iteration_us = 0.0;
                for (std::size_t i = 0; i < probe_repetitions; ++i) sum_iteration_us += strategy->run_once().e2e_us;
                const auto probe_end = std::chrono::steady_clock::now();
                const double probe_wall_us = std::chrono::duration<double, std::micro>(probe_end - probe_start).count();
                std::cout << "{\"event\":\"probe_done\",\"repetitions\":" << probe_repetitions
                          << ",\"batch_wall_us\":" << probe_wall_us
                          << ",\"mean_iteration_us\":" << (sum_iteration_us / static_cast<double>(probe_repetitions))
                          << "}" << std::endl;
                if (!std::getline(std::cin, command)) throw std::runtime_error("control pipe closed before TIMING");
            }
        }
        std::istringstream timing_cmd(command);
        std::size_t timing_repetitions = 0;
        timing_cmd >> verb >> timing_repetitions;
        if (verb != "TIMING" || timing_repetitions == 0) {
            throw std::runtime_error("expected command: TIMING <positive repetitions>");
        }

        std::vector<prbench::IterationMetrics> metrics;
        metrics.reserve(timing_repetitions);
        const auto timing_start = std::chrono::steady_clock::now();
        for (std::size_t i = 0; i < timing_repetitions; ++i) {
            metrics.push_back(strategy->run_once());
        }
        const auto timing_end = std::chrono::steady_clock::now();
        const double timing_wall_us =
            std::chrono::duration<double, std::micro>(timing_end - timing_start).count();
        const double timing_mean_us = timing_wall_us / static_cast<double>(timing_repetitions);
        std::cout << "{\"event\":\"timing_done\",\"repetitions\":" << timing_repetitions
                  << ",\"batch_wall_us\":" << timing_wall_us
                  << ",\"mean_iteration_us\":" << timing_mean_us << "}" << std::endl;

        if (!std::getline(std::cin, command)) {
            throw std::runtime_error("control pipe closed before ENERGY/DUMP");
        }

        std::size_t energy_repetitions = 0;
        if (command == "DUMP") {
            // Timing-only runs (for example parameter tuning) intentionally skip the
            // additional energy batch.  This avoids unnecessary work and thermal load.
        } else {
            std::istringstream energy_cmd(command);
            energy_cmd >> verb >> energy_repetitions;
            if (verb != "ENERGY" || energy_repetitions == 0) {
                throw std::runtime_error("expected command: ENERGY <positive repetitions> or DUMP");
            }

            prbench::Value energy_result = prbench::Value::identity(cfg.dtype, cfg.operation);
            const auto energy_start = std::chrono::steady_clock::now();
            for (std::size_t i = 0; i < energy_repetitions; ++i) {
                energy_result = strategy->run_once().result;
            }
            const auto energy_end = std::chrono::steady_clock::now();
            const double energy_wall_us =
                std::chrono::duration<double, std::micro>(energy_end - energy_start).count();

            // Emitted immediately after the energy batch. Per-repetition timing
            // serialization happens later so file/pipe I/O is outside the measured batch.
            std::cout << "{\"event\":\"measure_done\",\"repetitions\":" << energy_repetitions
                      << ",\"batch_wall_us\":" << energy_wall_us
                      << ",\"result\":" << energy_result.json_literal() << "}" << std::endl;

            if (!std::getline(std::cin, command) || command != "DUMP") {
                throw std::runtime_error("expected command: DUMP");
            }
        }

        for (std::size_t i = 0; i < metrics.size(); ++i) emit_result(i + 1, metrics[i], cfg);
        std::cout << "{\"event\":\"done\",\"timing_repetitions\":" << timing_repetitions
                  << ",\"energy_repetitions\":" << energy_repetitions << "}" << std::endl;
        return 0;
    } catch (const std::exception& exc) {
        std::cout << "{\"event\":\"error\",\"message\":\"" << json_escape(exc.what()) << "\"}"
                  << std::endl;
        std::cerr << "prbench-worker error: " << exc.what() << std::endl;
        return 1;
    }
}
