#include "prbench/cli.hpp"

#include <algorithm>
#include <cstdlib>
#include <stdexcept>
#include <string_view>

namespace prbench {
namespace {

std::vector<int> parse_int_list(const std::string& value) {
    std::vector<int> result;
    std::size_t start = 0;
    while (start < value.size()) {
        const auto end = value.find(',', start);
        const auto token = value.substr(start, end == std::string::npos ? std::string::npos : end - start);
        if (!token.empty()) result.push_back(std::stoi(token));
        if (end == std::string::npos) break;
        start = end + 1;
    }
    return result;
}

std::string require_value(int& i, int argc, char** argv, const std::string& arg) {
    if (i + 1 >= argc) throw std::invalid_argument("missing value after " + arg);
    return argv[++i];
}

}  // namespace

WorkerConfig parse_cli(int argc, char** argv) {
    WorkerConfig cfg;
    if (argc == 2 && std::string_view(argv[1]) == "--self-test") {
        cfg.self_test = true;
        return cfg;
    }
    if (argc == 2 && std::string_view(argv[1]) == "--build-info") {
        cfg.build_info = true;
        return cfg;
    }
    for (int i = 1; i < argc; ++i) {
        const std::string arg = argv[i];
        if (arg == "--probe-cpu-types") cfg.probe_cpu_types = true;
        else if (arg == "--probe-cpus") cfg.probe_cpus = parse_int_list(require_value(i, argc, argv, arg));
        else if (arg == "--dataset") cfg.dataset_path = require_value(i, argc, argv, arg);
        else if (arg == "--dtype") cfg.dtype = parse_data_type(require_value(i, argc, argv, arg));
        else if (arg == "--operation") cfg.operation = parse_reduction_operation(require_value(i, argc, argv, arg));
        else if (arg == "--count") cfg.count = std::stoull(require_value(i, argc, argv, arg));
        else if (arg == "--scheduler") cfg.scheduler = parse_scheduler(require_value(i, argc, argv, arg));
        else if (arg == "--cpu-backend") cfg.cpu_backend = parse_cpu_backend(require_value(i, argc, argv, arg));
        else if (arg == "--gpu-backend") cfg.gpu_backend = parse_gpu_backend(require_value(i, argc, argv, arg));
        else if (arg == "--transfer-policy") cfg.transfer_policy = parse_transfer_policy(require_value(i, argc, argv, arg));
        else if (arg == "--gpus") cfg.gpu_ids = parse_int_list(require_value(i, argc, argv, arg));
        else if (arg == "--cpu-affinity") cfg.cpu_affinity = parse_int_list(require_value(i, argc, argv, arg));
        else if (arg == "--gpu-worker-cpus") cfg.gpu_worker_cpus = parse_int_list(require_value(i, argc, argv, arg));
        else if (arg == "--cpu-threads") cfg.cpu_threads = std::stoi(require_value(i, argc, argv, arg));
        else if (arg == "--warmup-runs") cfg.warmup_runs = std::stoi(require_value(i, argc, argv, arg));
        else if (arg == "--calibration-repetitions") cfg.calibration_repetitions = std::stoi(require_value(i, argc, argv, arg));
        else if (arg == "--cache-rotation-target-bytes") cfg.cache_rotation_target_bytes = std::stoull(require_value(i, argc, argv, arg));
        else if (arg == "--cache-rotation-max-replicas") cfg.cache_rotation_max_replicas = std::stoull(require_value(i, argc, argv, arg));
        else if (arg == "--block-size") cfg.block_size = std::stoi(require_value(i, argc, argv, arg));
        else if (arg == "--chunk-size") cfg.chunk_size = std::stoull(require_value(i, argc, argv, arg));
        else if (arg == "--min-chunk-size") cfg.min_chunk_size = std::stoull(require_value(i, argc, argv, arg));
        else if (arg == "--max-chunk-size") cfg.max_chunk_size = std::stoull(require_value(i, argc, argv, arg));
        else if (arg == "--guided-factor") cfg.guided_factor = std::stod(require_value(i, argc, argv, arg));
        else if (arg == "--target-chunk-ms") cfg.target_chunk_ms = std::stod(require_value(i, argc, argv, arg));
        else if (arg == "--ema-alpha") cfg.ema_alpha = std::stod(require_value(i, argc, argv, arg));
        else if (arg == "--pipeline-streams") cfg.pipeline_streams = std::stoi(require_value(i, argc, argv, arg));
        else if (arg == "--pipeline-chunks") cfg.pipeline_chunks = std::stoi(require_value(i, argc, argv, arg));
        else if (arg == "--pipeline-chunk-elements") cfg.pipeline_chunk_elements = std::stoull(require_value(i, argc, argv, arg));
        else throw std::invalid_argument("unknown argument: " + arg);
    }

    if (cfg.build_info) return cfg;

    if (cfg.probe_cpu_types) {
        if (cfg.probe_cpus.empty()) throw std::invalid_argument("--probe-cpus is required with --probe-cpu-types");
        return cfg;
    }

    if (cfg.dataset_path.empty()) throw std::invalid_argument("--dataset is required");
    if (cfg.count == 0) throw std::invalid_argument("--count must be positive");
    if (cfg.cpu_threads < 1) throw std::invalid_argument("--cpu-threads must be positive");
    if (cfg.warmup_runs < 1) throw std::invalid_argument("--warmup-runs must be positive");
    if (cfg.calibration_repetitions < 1 || cfg.calibration_repetitions > 100) {
        throw std::invalid_argument("--calibration-repetitions must be in [1,100]");
    }
    if (cfg.cache_rotation_max_replicas < 1 || cfg.cache_rotation_max_replicas > 1024) {
        throw std::invalid_argument("--cache-rotation-max-replicas must be in [1,1024]");
    }
    if (cfg.block_size < 32 || cfg.block_size > 1024 || (cfg.block_size & (cfg.block_size - 1)) != 0) {
        throw std::invalid_argument("--block-size must be a power of two in [32, 1024]");
    }
    if (cfg.pipeline_streams < 1 || cfg.pipeline_chunks < cfg.pipeline_streams) {
        throw std::invalid_argument("invalid pipeline_streams/pipeline_chunks");
    }
    if (!(cfg.ema_alpha > 0.0 && cfg.ema_alpha <= 1.0)) {
        throw std::invalid_argument("--ema-alpha must be in (0,1]");
    }
    if (cfg.guided_factor <= 0.0 || cfg.target_chunk_ms <= 0.0) {
        throw std::invalid_argument("guided_factor and target_chunk_ms must be positive");
    }
    return cfg;
}

}  // namespace prbench
