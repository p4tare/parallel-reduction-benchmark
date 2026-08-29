#include "prbench/strategy.hpp"

#include <algorithm>
#include <atomic>
#include <chrono>
#include <cmath>
#include <exception>
#include <memory>
#include <mutex>
#include <numeric>
#include <stdexcept>
#include <thread>
#include <utility>

#include "prbench/affinity.hpp"
#include "prbench/cpu_backend.hpp"
#include "prbench/gpu_backend.hpp"

namespace prbench {

std::size_t dynamic_calibration_elements(
    SchedulerKind kind,
    std::size_t dataset_count,
    std::size_t min_chunk_size,
    std::size_t max_chunk_size
) {
    if (kind != SchedulerKind::DynamicAdaptive || dataset_count == 0) return 0;
    const std::size_t desired = std::max<std::size_t>(min_chunk_size, std::size_t{1} << 20);
    return std::min({dataset_count, max_chunk_size, desired});
}

namespace {

using clock_type = std::chrono::steady_clock;

struct Range {
    std::size_t offset{0};
    std::size_t count{0};
};

struct LinearModel {
    double intercept_us{0.0};
    double slope_us_per_element{1.0};

    double predict(std::size_t n) const noexcept {
        return intercept_us + slope_us_per_element * static_cast<double>(n);
    }

    double capacity(double time_us) const noexcept {
        if (time_us <= intercept_us || slope_us_per_element <= 0.0) return 0.0;
        return (time_us - intercept_us) / slope_us_per_element;
    }
};

double median(std::vector<double> values) {
    if (values.empty()) return 0.0;
    std::sort(values.begin(), values.end());
    const std::size_t mid = values.size() / 2;
    if (values.size() % 2) return values[mid];
    return 0.5 * (values[mid - 1] + values[mid]);
}

void accumulate_device_metrics(DeviceMetrics& dst, const DeviceMetrics& src) {
    dst.device_id = src.device_id;
    dst.h2d_us += src.h2d_us;
    dst.kernel_us += src.kernel_us;
    dst.d2h_us += src.d2h_us;
    dst.device_overhead_us += src.device_overhead_us;
    dst.total_us += src.total_us;
    dst.chunks += src.chunks;
    dst.elements += src.elements;
}

LinearModel fit_linear_model(const std::vector<std::pair<std::size_t, double>>& samples) {
    if (samples.empty()) return {};
    if (samples.size() == 1) {
        const auto [n, t] = samples.front();
        return LinearModel{0.0, std::max(1e-12, t / std::max<std::size_t>(1, n))};
    }
    double sum_x = 0.0, sum_y = 0.0, sum_xx = 0.0, sum_xy = 0.0;
    for (const auto& [n, t] : samples) {
        const double x = static_cast<double>(n);
        sum_x += x;
        sum_y += t;
        sum_xx += x * x;
        sum_xy += x * t;
    }
    const double m = static_cast<double>(samples.size());
    const double denom = m * sum_xx - sum_x * sum_x;
    double slope = denom != 0.0 ? (m * sum_xy - sum_x * sum_y) / denom : 0.0;
    double intercept = (sum_y - slope * sum_x) / m;
    slope = std::max(slope, 1e-12);
    intercept = std::max(0.0, intercept);
    return LinearModel{intercept, slope};
}

std::vector<Range> equal_partition(std::size_t total, std::size_t engines) {
    std::vector<Range> ranges(engines);
    const std::size_t base = engines ? total / engines : 0;
    const std::size_t rem = engines ? total % engines : 0;
    std::size_t offset = 0;
    for (std::size_t i = 0; i < engines; ++i) {
        const std::size_t count = base + (i < rem ? 1 : 0);
        ranges[i] = Range{offset, count};
        offset += count;
    }
    return ranges;
}

std::vector<Range> model_partition(std::size_t total, const std::vector<LinearModel>& models) {
    if (models.empty()) return {};
    double lo = 0.0;
    double hi = 0.0;
    for (const auto& model : models) hi = std::max(hi, model.predict(total));
    hi = std::max(hi, 1.0);
    for (int iter = 0; iter < 80; ++iter) {
        const double mid = 0.5 * (lo + hi);
        double capacity = 0.0;
        for (const auto& model : models) capacity += model.capacity(mid);
        if (capacity >= static_cast<double>(total)) hi = mid;
        else lo = mid;
    }
    std::vector<double> raw;
    raw.reserve(models.size());
    double sum = 0.0;
    for (const auto& model : models) {
        const double c = std::max(0.0, model.capacity(hi));
        raw.push_back(c);
        sum += c;
    }
    if (sum <= 0.0) return equal_partition(total, models.size());

    std::vector<Range> ranges(models.size());
    std::size_t assigned = 0;
    std::size_t offset = 0;
    for (std::size_t i = 0; i < models.size(); ++i) {
        std::size_t count = (i + 1 == models.size())
            ? total - assigned
            : static_cast<std::size_t>(std::floor(raw[i] / sum * static_cast<double>(total)));
        ranges[i] = Range{offset, count};
        offset += count;
        assigned += count;
    }
    return ranges;
}

class StrategyBase : public IReductionStrategy {
public:
    double warmup_median_us() const noexcept override { return warmup_median_us_; }
    const PrepareMetrics& prepare_metrics() const noexcept override { return prepare_metrics_; }

protected:
    PrepareMetrics prepare_metrics_;
    void run_warmups(int runs) {
        std::vector<double> times;
        times.reserve(static_cast<std::size_t>(runs));
        for (int i = 0; i < runs; ++i) {
            times.push_back(run_once().e2e_us);
        }
        warmup_median_us_ = median(std::move(times));
    }

private:
    double warmup_median_us_{0.0};
};

class CpuOnlyStrategy final : public StrategyBase {
public:
    CpuOnlyStrategy(const WorkerConfig& cfg, Dataset& dataset) : cfg_(cfg), dataset_(dataset) {}

    void prepare(int warmup_runs) override {
        prepare_metrics_.partition = {{"cpu", 0, -1, 0, dataset_.count()}};
        run_warmups(warmup_runs);
    }

    IterationMetrics run_once() override {
        dataset_.advance_replica();
        const auto begin = clock_type::now();
        auto cpu = reduce_cpu(dataset_.data(), dataset_.count(), dataset_.dtype(), cfg_.cpu_backend, cfg_.cpu_threads, cfg_.operation);
        const auto end = clock_type::now();
        IterationMetrics out;
        out.result = cpu.result;
        out.cpu.compute_us = cpu.compute_us;
        out.cpu.chunks = 1;
        out.cpu.elements = dataset_.count();
        out.e2e_us = std::chrono::duration<double, std::micro>(end - begin).count();
        return out;
    }

private:
    const WorkerConfig& cfg_;
    Dataset& dataset_;
};

class GpuOnlyStrategy final : public StrategyBase {
public:
    GpuOnlyStrategy(const WorkerConfig& cfg, Dataset& dataset) : cfg_(cfg), dataset_(dataset) {
        if (cfg_.gpu_ids.size() != 1) throw std::invalid_argument("gpu_only requires exactly one GPU");
        GpuReducerConfig gc;
        gc.device_id = cfg_.gpu_ids.front();
        gc.backend = cfg_.gpu_backend;
        gc.transfer_policy = cfg_.transfer_policy;
        gc.dtype = cfg_.dtype;
        gc.operation = cfg_.operation;
        gc.max_elements = dataset_.count();
        gc.block_size = cfg_.block_size;
        gc.pipeline_streams = cfg_.pipeline_streams;
        gc.pipeline_chunks = cfg_.pipeline_chunks;
        gc.pipeline_chunk_elements = cfg_.pipeline_chunk_elements;
        gpu_ = make_gpu_reducer(gc);
    }

    void prepare(int warmup_runs) override {
        prepare_metrics_.partition = {{"gpu", 0, cfg_.gpu_ids.front(), 0, dataset_.count()}};
        if (cfg_.transfer_policy == TransferPolicy::DeviceResident) {
            // Explicit first-use sample: includes one-time H2D residency establishment
            // plus one reduction. Steady-state TIMING begins only after this sample and
            // ordinary warm-ups, so first-use vs resident steady-state can be reported.
            dataset_.advance_replica();
            const auto start = clock_type::now();
            (void)gpu_->reduce(dataset_.data(), dataset_.count());
            const auto end = clock_type::now();
            const double elapsed =
                std::chrono::duration<double, std::micro>(end - start).count();
            prepare_metrics_.calibration_samples.push_back({
                "gpu_device_resident_first_use",
                0,
                cfg_.gpu_ids.front(),
                dataset_.count(),
                elapsed,
                dataset_.count() / std::max(1e-12, elapsed * 1e-6),
            });
        }
        run_warmups(warmup_runs);
    }

    IterationMetrics run_once() override {
        if (cfg_.transfer_policy != TransferPolicy::DeviceResident) {
            dataset_.advance_replica();
        }
        const auto begin = clock_type::now();
        auto partial = gpu_->reduce(dataset_.data(), dataset_.count());
        const auto end = clock_type::now();
        IterationMetrics out;
        out.result = partial.result;
        out.gpus.push_back(partial.device);
        out.e2e_us = std::chrono::duration<double, std::micro>(end - begin).count();
        return out;
    }

private:
    const WorkerConfig& cfg_;
    Dataset& dataset_;
    std::unique_ptr<IGpuReducer> gpu_;
};

class MultiGpuStrategy final : public StrategyBase {
public:
    MultiGpuStrategy(const WorkerConfig& cfg, Dataset& dataset, bool profiled)
        : cfg_(cfg), dataset_(dataset), profiled_(profiled) {
        if (cfg_.gpu_ids.size() < 2) throw std::invalid_argument("multi-GPU strategy requires at least two GPUs");
    }

    void prepare(int warmup_runs) override {
        ranges_ = profiled_ ? profile_and_partition() : equal_partition(dataset_.count(), cfg_.gpu_ids.size());
        prepare_metrics_.partition.clear();
        for (std::size_t i = 0; i < ranges_.size(); ++i) {
            prepare_metrics_.partition.push_back({"gpu", static_cast<int>(i), cfg_.gpu_ids[i], ranges_[i].offset, ranges_[i].count});
        }
        create_reducers();
        run_warmups(warmup_runs);
    }

    IterationMetrics run_once() override {
        dataset_.advance_replica();
        const auto e2e_start = clock_type::now();
        std::vector<PartialResult> gpu_results(cfg_.gpu_ids.size());
        std::vector<std::exception_ptr> errors(cfg_.gpu_ids.size());
        std::vector<std::thread> threads;
        threads.reserve(cfg_.gpu_ids.size());
        for (std::size_t i = 0; i < cfg_.gpu_ids.size(); ++i) {
            threads.emplace_back([&, i] {
                try {
                    if (i < cfg_.gpu_worker_cpus.size()) pin_current_thread(cfg_.gpu_worker_cpus[i]);
                    const auto& range = ranges_[i];
                    gpu_results[i] = gpus_[i]->reduce(dataset_.offset_ptr(range.offset), range.count);
                } catch (...) {
                    errors[i] = std::current_exception();
                }
            });
        }
        for (auto& thread : threads) thread.join();
        for (const auto& error : errors) if (error) std::rethrow_exception(error);

        const auto merge_start = clock_type::now();
        Value result = Value::identity(dataset_.dtype(), cfg_.operation);
        for (const auto& partial : gpu_results) result.combine(partial.result, dataset_.dtype(), cfg_.operation);
        const auto merge_end = clock_type::now();
        const auto e2e_end = clock_type::now();

        IterationMetrics out;
        out.result = result;
        for (const auto& partial : gpu_results) out.gpus.push_back(partial.device);
        out.merge_us = std::chrono::duration<double, std::micro>(merge_end - merge_start).count();
        out.e2e_us = std::chrono::duration<double, std::micro>(e2e_end - e2e_start).count();
        return out;
    }

private:
    std::vector<Range> profile_and_partition() {
        std::vector<std::size_t> sizes;
        for (std::size_t candidate : {std::size_t{262144}, std::size_t{1048576}, std::size_t{4194304}}) {
            const auto value = std::min(candidate, dataset_.count());
            if (value > 0 && std::find(sizes.begin(), sizes.end(), value) == sizes.end()) sizes.push_back(value);
        }
        if (sizes.empty()) sizes.push_back(dataset_.count());
        const std::size_t max_sample = *std::max_element(sizes.begin(), sizes.end());
        std::vector<LinearModel> models;
        models.reserve(cfg_.gpu_ids.size());
        for (std::size_t i = 0; i < cfg_.gpu_ids.size(); ++i) {
            std::unique_ptr<ScopedThreadAffinity> affinity;
            if (i < cfg_.gpu_worker_cpus.size()) {
                affinity = std::make_unique<ScopedThreadAffinity>(cfg_.gpu_worker_cpus[i]);
            }
            GpuReducerConfig gc;
            gc.device_id = cfg_.gpu_ids[i];
            gc.backend = cfg_.gpu_backend;
            gc.transfer_policy = cfg_.transfer_policy;
            gc.dtype = cfg_.dtype;
            gc.operation = cfg_.operation;
            gc.max_elements = max_sample;
            gc.block_size = cfg_.block_size;
            gc.pipeline_streams = cfg_.pipeline_streams;
            gc.pipeline_chunks = cfg_.pipeline_chunks;
        gc.pipeline_chunk_elements = cfg_.pipeline_chunk_elements;
            auto gpu = make_gpu_reducer(gc);
            std::vector<std::pair<std::size_t, double>> samples;
            for (auto n : sizes) {
                dataset_.advance_replica();
                (void)gpu->reduce(dataset_.data(), n);
                std::vector<double> times;
                times.reserve(static_cast<std::size_t>(cfg_.calibration_repetitions));
                for (int rep = 0; rep < cfg_.calibration_repetitions; ++rep) {
                    dataset_.advance_replica();
                    auto sample = gpu->reduce(dataset_.data(), n);
                    times.push_back(sample.device.total_us);
                }
                const double elapsed = median(std::move(times));
                samples.emplace_back(n, elapsed);
                prepare_metrics_.calibration_samples.push_back({
                    "gpu", static_cast<int>(i), cfg_.gpu_ids[i], n, elapsed,
                    static_cast<double>(n) / std::max(elapsed, 1e-6) * 1e6
                });
            }
            const auto model = fit_linear_model(samples);
            models.push_back(model);
            prepare_metrics_.linear_models.push_back({
                "gpu", static_cast<int>(i), cfg_.gpu_ids[i], model.intercept_us, model.slope_us_per_element
            });
        }
        return model_partition(dataset_.count(), models);
    }

    void create_reducers() {
        gpus_.clear();
        for (std::size_t i = 0; i < cfg_.gpu_ids.size(); ++i) {
            std::unique_ptr<ScopedThreadAffinity> affinity;
            if (i < cfg_.gpu_worker_cpus.size()) {
                affinity = std::make_unique<ScopedThreadAffinity>(cfg_.gpu_worker_cpus[i]);
            }
            GpuReducerConfig gc;
            gc.device_id = cfg_.gpu_ids[i];
            gc.backend = cfg_.gpu_backend;
            gc.transfer_policy = cfg_.transfer_policy;
            gc.dtype = cfg_.dtype;
            gc.operation = cfg_.operation;
            gc.max_elements = std::max<std::size_t>(1, ranges_[i].count);
            gc.block_size = cfg_.block_size;
            gc.pipeline_streams = cfg_.pipeline_streams;
            gc.pipeline_chunks = cfg_.pipeline_chunks;
        gc.pipeline_chunk_elements = cfg_.pipeline_chunk_elements;
            gpus_.push_back(make_gpu_reducer(gc));
        }
    }

    const WorkerConfig& cfg_;
    Dataset& dataset_;
    bool profiled_;
    std::vector<Range> ranges_;
    std::vector<std::unique_ptr<IGpuReducer>> gpus_;
};

class StaticHybridStrategy final : public StrategyBase {
public:
    StaticHybridStrategy(const WorkerConfig& cfg, Dataset& dataset, bool profiled)
        : cfg_(cfg), dataset_(dataset), profiled_(profiled) {
        if (cfg_.gpu_ids.empty()) throw std::invalid_argument("hybrid strategy requires at least one GPU");
    }

    void prepare(int warmup_runs) override {
        if (profiled_) ranges_ = profile_and_partition();
        else ranges_ = equal_partition(dataset_.count(), 1 + cfg_.gpu_ids.size());
        prepare_metrics_.partition.clear();
        prepare_metrics_.partition.push_back({"cpu", 0, -1, ranges_[0].offset, ranges_[0].count});
        for (std::size_t i = 0; i < cfg_.gpu_ids.size(); ++i) {
            prepare_metrics_.partition.push_back({"gpu", static_cast<int>(i), cfg_.gpu_ids[i], ranges_[i + 1].offset, ranges_[i + 1].count});
        }
        create_final_gpu_reducers();
        run_warmups(warmup_runs);
    }

    IterationMetrics run_once() override {
        dataset_.advance_replica();
        const auto e2e_start = clock_type::now();
        std::vector<PartialResult> gpu_results(cfg_.gpu_ids.size());
        std::vector<std::exception_ptr> errors(cfg_.gpu_ids.size());
        std::vector<std::thread> threads;
        threads.reserve(cfg_.gpu_ids.size());

        for (std::size_t i = 0; i < cfg_.gpu_ids.size(); ++i) {
            threads.emplace_back([&, i] {
                try {
                    if (i < cfg_.gpu_worker_cpus.size()) pin_current_thread(cfg_.gpu_worker_cpus[i]);
                    const auto& range = ranges_[i + 1];
                    gpu_results[i] = gpus_[i]->reduce(dataset_.offset_ptr(range.offset), range.count);
                } catch (...) {
                    errors[i] = std::current_exception();
                }
            });
        }

        const auto& cpu_range = ranges_.front();
        CpuReductionResult cpu{Value::identity(dataset_.dtype(), cfg_.operation), 0.0};
        if (cpu_range.count > 0) {
            cpu = reduce_cpu(
                dataset_.offset_ptr(cpu_range.offset), cpu_range.count, dataset_.dtype(),
                cfg_.cpu_backend, cfg_.cpu_threads, cfg_.operation
            );
        }
        for (auto& thread : threads) thread.join();
        for (const auto& error : errors) if (error) std::rethrow_exception(error);

        const auto merge_start = clock_type::now();
        Value result = cpu.result;
        for (const auto& partial : gpu_results) result.combine(partial.result, dataset_.dtype(), cfg_.operation);
        const auto merge_end = clock_type::now();
        const auto e2e_end = clock_type::now();

        IterationMetrics out;
        out.result = result;
        out.cpu.compute_us = cpu.compute_us;
        out.cpu.chunks = cpu_range.count ? 1 : 0;
        out.cpu.elements = cpu_range.count;
        for (const auto& partial : gpu_results) out.gpus.push_back(partial.device);
        out.merge_us = std::chrono::duration<double, std::micro>(merge_end - merge_start).count();
        out.e2e_us = std::chrono::duration<double, std::micro>(e2e_end - e2e_start).count();
        return out;
    }

private:
    std::vector<Range> profile_and_partition() {
        std::vector<std::size_t> sizes;
        for (std::size_t candidate : {std::size_t{262144}, std::size_t{1048576}, std::size_t{4194304}}) {
            const auto value = std::min(candidate, dataset_.count());
            if (value > 0 && std::find(sizes.begin(), sizes.end(), value) == sizes.end()) sizes.push_back(value);
        }
        if (sizes.empty()) sizes.push_back(dataset_.count());

        std::vector<LinearModel> models;
        std::vector<std::pair<std::size_t, double>> cpu_samples;
        for (auto n : sizes) {
            dataset_.advance_replica();
            (void)reduce_cpu(dataset_.data(), n, dataset_.dtype(), cfg_.cpu_backend, cfg_.cpu_threads, cfg_.operation);
            std::vector<double> times;
            times.reserve(static_cast<std::size_t>(cfg_.calibration_repetitions));
            for (int rep = 0; rep < cfg_.calibration_repetitions; ++rep) {
                dataset_.advance_replica();
                auto sample = reduce_cpu(dataset_.data(), n, dataset_.dtype(), cfg_.cpu_backend, cfg_.cpu_threads, cfg_.operation);
                times.push_back(sample.compute_us);
            }
            const double elapsed = median(std::move(times));
            cpu_samples.emplace_back(n, elapsed);
            prepare_metrics_.calibration_samples.push_back({
                "cpu", 0, -1, n, elapsed,
                static_cast<double>(n) / std::max(elapsed, 1e-6) * 1e6
            });
        }
        const auto cpu_model = fit_linear_model(cpu_samples);
        models.push_back(cpu_model);
        prepare_metrics_.linear_models.push_back({"cpu", 0, -1, cpu_model.intercept_us, cpu_model.slope_us_per_element});

        const std::size_t max_sample = *std::max_element(sizes.begin(), sizes.end());
        for (std::size_t i = 0; i < cfg_.gpu_ids.size(); ++i) {
            std::unique_ptr<ScopedThreadAffinity> affinity;
            if (i < cfg_.gpu_worker_cpus.size()) {
                affinity = std::make_unique<ScopedThreadAffinity>(cfg_.gpu_worker_cpus[i]);
            }
            GpuReducerConfig gc;
            gc.device_id = cfg_.gpu_ids[i];
            gc.backend = cfg_.gpu_backend;
            gc.transfer_policy = cfg_.transfer_policy;
            gc.dtype = cfg_.dtype;
            gc.operation = cfg_.operation;
            gc.max_elements = max_sample;
            gc.block_size = cfg_.block_size;
            gc.pipeline_streams = cfg_.pipeline_streams;
            gc.pipeline_chunks = cfg_.pipeline_chunks;
        gc.pipeline_chunk_elements = cfg_.pipeline_chunk_elements;
            auto gpu = make_gpu_reducer(gc);
            std::vector<std::pair<std::size_t, double>> samples;
            for (auto n : sizes) {
                dataset_.advance_replica();
                (void)gpu->reduce(dataset_.data(), n);
                std::vector<double> times;
                times.reserve(static_cast<std::size_t>(cfg_.calibration_repetitions));
                for (int rep = 0; rep < cfg_.calibration_repetitions; ++rep) {
                    dataset_.advance_replica();
                    auto sample = gpu->reduce(dataset_.data(), n);
                    times.push_back(sample.device.total_us);
                }
                const double elapsed = median(std::move(times));
                samples.emplace_back(n, elapsed);
                prepare_metrics_.calibration_samples.push_back({
                    "gpu", static_cast<int>(i), cfg_.gpu_ids[i], n, elapsed,
                    static_cast<double>(n) / std::max(elapsed, 1e-6) * 1e6
                });
            }
            const auto model = fit_linear_model(samples);
            models.push_back(model);
            prepare_metrics_.linear_models.push_back({
                "gpu", static_cast<int>(i), cfg_.gpu_ids[i], model.intercept_us, model.slope_us_per_element
            });
        }
        return model_partition(dataset_.count(), models);
    }

    void create_final_gpu_reducers() {
        gpus_.clear();
        for (std::size_t i = 0; i < cfg_.gpu_ids.size(); ++i) {
            std::unique_ptr<ScopedThreadAffinity> affinity;
            if (i < cfg_.gpu_worker_cpus.size()) {
                affinity = std::make_unique<ScopedThreadAffinity>(cfg_.gpu_worker_cpus[i]);
            }
            GpuReducerConfig gc;
            gc.device_id = cfg_.gpu_ids[i];
            gc.backend = cfg_.gpu_backend;
            gc.transfer_policy = cfg_.transfer_policy;
            gc.dtype = cfg_.dtype;
            gc.operation = cfg_.operation;
            gc.max_elements = std::max<std::size_t>(1, ranges_[i + 1].count);
            gc.block_size = cfg_.block_size;
            gc.pipeline_streams = cfg_.pipeline_streams;
            gc.pipeline_chunks = cfg_.pipeline_chunks;
        gc.pipeline_chunk_elements = cfg_.pipeline_chunk_elements;
            gpus_.push_back(make_gpu_reducer(gc));
        }
    }

    const WorkerConfig& cfg_;
    Dataset& dataset_;
    bool profiled_;
    std::vector<Range> ranges_;
    std::vector<std::unique_ptr<IGpuReducer>> gpus_;
};

class DynamicHybridStrategy final : public StrategyBase {
public:
    DynamicHybridStrategy(const WorkerConfig& cfg, Dataset& dataset, SchedulerKind kind)
        : cfg_(cfg), dataset_(dataset), kind_(kind) {
        if (cfg_.gpu_ids.empty()) throw std::invalid_argument("dynamic hybrid strategy requires at least one GPU");
    }

    void prepare(int warmup_runs) override {
        // Fixed and guided schedulers do not use throughput estimates at all.  Earlier
        // versions nevertheless ran a 1 Mi-element calibration on every dynamic
        // scheduler while the fixed-chunk GPU reducer could have been allocated for a
        // smaller chunk (for example 262144 elements).  That violated the reducer's
        // capacity contract before the first timed iteration.  Calibrate only the
        // adaptive scheduler and size its reducer for both the runtime maximum chunk and
        // the calibration sample.
        const std::size_t runtime_max_chunk = kind_ == SchedulerKind::DynamicFixed
            ? cfg_.chunk_size
            : cfg_.max_chunk_size;
        const std::size_t calibration_elements = dynamic_calibration_elements(
            kind_, dataset_.count(), cfg_.min_chunk_size, cfg_.max_chunk_size
        );
        const std::size_t reducer_capacity = std::max(runtime_max_chunk, calibration_elements);

        for (int gpu_id : cfg_.gpu_ids) {
            GpuReducerConfig gc;
            gc.device_id = gpu_id;
            gc.backend = cfg_.gpu_backend;
            gc.transfer_policy = TransferPolicy::Sync;
            gc.dtype = cfg_.dtype;
            gc.operation = cfg_.operation;
            gc.max_elements = std::max<std::size_t>(1, reducer_capacity);
            gc.block_size = cfg_.block_size;
            gpus_.push_back(make_gpu_reducer(gc));
        }
        if (kind_ == SchedulerKind::DynamicAdaptive) initialize_throughput(calibration_elements);
        run_warmups(warmup_runs);
    }

    IterationMetrics run_once() override {
        dataset_.advance_replica();
        next_.store(0, std::memory_order_relaxed);
        scheduler_accum_us_.store(0.0, std::memory_order_relaxed);
        const auto e2e_start = clock_type::now();

        Value cpu_value = Value::identity(dataset_.dtype(), cfg_.operation);
        CpuMetrics cpu_metrics;
        std::vector<Value> gpu_values(cfg_.gpu_ids.size(), Value::identity(dataset_.dtype(), cfg_.operation));
        std::vector<DeviceMetrics> gpu_metrics(cfg_.gpu_ids.size());
        std::vector<std::exception_ptr> errors(cfg_.gpu_ids.size());
        std::vector<std::thread> threads;
        threads.reserve(cfg_.gpu_ids.size());

        for (std::size_t i = 0; i < cfg_.gpu_ids.size(); ++i) {
            gpu_metrics[i].device_id = cfg_.gpu_ids[i];
            threads.emplace_back([&, i] {
                try {
                    if (i < cfg_.gpu_worker_cpus.size()) pin_current_thread(cfg_.gpu_worker_cpus[i]);
                    const std::size_t worker_id = i + 1;
                    while (true) {
                        const auto claim_start = clock_type::now();
                        const Range range = claim(worker_id);
                        const auto claim_end = clock_type::now();
                        scheduler_accum_us_.fetch_add(
                            std::chrono::duration<double, std::micro>(claim_end - claim_start).count(),
                            std::memory_order_relaxed
                        );
                        if (range.count == 0) break;
                        const auto chunk_start = clock_type::now();
                        auto partial = gpus_[i]->reduce(dataset_.offset_ptr(range.offset), range.count);
                        const auto chunk_end = clock_type::now();
                        gpu_values[i].combine(partial.result, dataset_.dtype(), cfg_.operation);
                        accumulate_device_metrics(gpu_metrics[i], partial.device);
                        update_throughput(
                            worker_id, range.count,
                            std::chrono::duration<double, std::micro>(chunk_end - chunk_start).count()
                        );
                    }
                } catch (...) {
                    errors[i] = std::current_exception();
                }
            });
        }

        while (true) {
            const auto claim_start = clock_type::now();
            const Range range = claim(0);
            const auto claim_end = clock_type::now();
            scheduler_accum_us_.fetch_add(
                std::chrono::duration<double, std::micro>(claim_end - claim_start).count(),
                std::memory_order_relaxed
            );
            if (range.count == 0) break;
            const auto chunk_start = clock_type::now();
            auto partial = reduce_cpu(
                dataset_.offset_ptr(range.offset), range.count, dataset_.dtype(),
                cfg_.cpu_backend, cfg_.cpu_threads, cfg_.operation
            );
            const auto chunk_end = clock_type::now();
            cpu_value.combine(partial.result, dataset_.dtype(), cfg_.operation);
            cpu_metrics.compute_us += partial.compute_us;
            cpu_metrics.chunks += 1;
            cpu_metrics.elements += range.count;
            update_throughput(
                0, range.count,
                std::chrono::duration<double, std::micro>(chunk_end - chunk_start).count()
            );
        }

        for (auto& thread : threads) thread.join();
        for (const auto& error : errors) if (error) std::rethrow_exception(error);

        const auto merge_start = clock_type::now();
        Value result = cpu_value;
        for (const auto& value : gpu_values) result.combine(value, dataset_.dtype(), cfg_.operation);
        const auto merge_end = clock_type::now();
        const auto e2e_end = clock_type::now();

        IterationMetrics out;
        out.result = result;
        out.cpu = cpu_metrics;
        out.gpus = gpu_metrics;
        out.scheduler_us = scheduler_accum_us_.load(std::memory_order_relaxed);
        out.merge_us = std::chrono::duration<double, std::micro>(merge_end - merge_start).count();
        out.e2e_us = std::chrono::duration<double, std::micro>(e2e_end - e2e_start).count();
        if (kind_ == SchedulerKind::DynamicAdaptive) {
            std::scoped_lock lock(throughput_mutex_);
            out.worker_throughput_elements_s = throughput_;
        }
        return out;
    }

private:
    void initialize_throughput(std::size_t sample_n) {
        if (sample_n == 0) throw std::logic_error("adaptive scheduler requires a positive calibration sample");
        throughput_.assign(1 + cfg_.gpu_ids.size(), 1.0);
        dataset_.advance_replica();
        (void)reduce_cpu(dataset_.data(), sample_n, dataset_.dtype(), cfg_.cpu_backend, cfg_.cpu_threads, cfg_.operation);
        std::vector<double> cpu_times;
        cpu_times.reserve(static_cast<std::size_t>(cfg_.calibration_repetitions));
        for (int rep = 0; rep < cfg_.calibration_repetitions; ++rep) {
            dataset_.advance_replica();
            auto cpu = reduce_cpu(dataset_.data(), sample_n, dataset_.dtype(), cfg_.cpu_backend, cfg_.cpu_threads, cfg_.operation);
            cpu_times.push_back(cpu.compute_us);
        }
        const double cpu_elapsed = median(std::move(cpu_times));
        throughput_[0] = static_cast<double>(sample_n) / std::max(cpu_elapsed, 1e-6) * 1e6;
        prepare_metrics_.calibration_samples.push_back({"cpu", 0, -1, sample_n, cpu_elapsed, throughput_[0]});
        for (std::size_t i = 0; i < gpus_.size(); ++i) {
            std::unique_ptr<ScopedThreadAffinity> affinity;
            if (i < cfg_.gpu_worker_cpus.size()) {
                affinity = std::make_unique<ScopedThreadAffinity>(cfg_.gpu_worker_cpus[i]);
            }
            dataset_.advance_replica();
            (void)gpus_[i]->reduce(dataset_.data(), sample_n);
            std::vector<double> gpu_times;
            gpu_times.reserve(static_cast<std::size_t>(cfg_.calibration_repetitions));
            for (int rep = 0; rep < cfg_.calibration_repetitions; ++rep) {
                dataset_.advance_replica();
                auto gpu = gpus_[i]->reduce(dataset_.data(), sample_n);
                gpu_times.push_back(gpu.device.total_us);
            }
            const double gpu_elapsed = median(std::move(gpu_times));
            throughput_[i + 1] = static_cast<double>(sample_n) / std::max(gpu_elapsed, 1e-6) * 1e6;
            prepare_metrics_.calibration_samples.push_back({
                "gpu", static_cast<int>(i), cfg_.gpu_ids[i], sample_n, gpu_elapsed, throughput_[i + 1]
            });
        }
        prepare_metrics_.initial_throughput_elements_s = throughput_;
    }

    Range claim(std::size_t worker_id) {
        while (true) {
            std::size_t current = next_.load(std::memory_order_relaxed);
            if (current >= dataset_.count()) return Range{dataset_.count(), 0};
            const std::size_t remaining = dataset_.count() - current;
            std::size_t chunk = cfg_.chunk_size;
            if (kind_ == SchedulerKind::DynamicGuided) {
                const double denom = cfg_.guided_factor * static_cast<double>(1 + cfg_.gpu_ids.size());
                chunk = static_cast<std::size_t>(std::ceil(static_cast<double>(remaining) / denom));
                chunk = std::clamp(chunk, cfg_.min_chunk_size, cfg_.max_chunk_size);
            } else if (kind_ == SchedulerKind::DynamicAdaptive) {
                double tput = 1.0;
                {
                    std::scoped_lock lock(throughput_mutex_);
                    tput = throughput_.at(worker_id);
                }
                chunk = static_cast<std::size_t>(
                    std::max(1.0, tput * (cfg_.target_chunk_ms / 1000.0))
                );
                chunk = std::clamp(chunk, cfg_.min_chunk_size, cfg_.max_chunk_size);
            }
            chunk = std::min(chunk, remaining);
            if (next_.compare_exchange_weak(current, current + chunk, std::memory_order_relaxed)) {
                return Range{current, chunk};
            }
        }
    }

    void update_throughput(std::size_t worker_id, std::size_t elements, double elapsed_us) {
        if (kind_ != SchedulerKind::DynamicAdaptive || elapsed_us <= 0.0) return;
        const double observed = static_cast<double>(elements) / elapsed_us * 1e6;
        std::scoped_lock lock(throughput_mutex_);
        double& current = throughput_.at(worker_id);
        current = cfg_.ema_alpha * observed + (1.0 - cfg_.ema_alpha) * current;
    }

    const WorkerConfig& cfg_;
    Dataset& dataset_;
    SchedulerKind kind_;
    std::vector<std::unique_ptr<IGpuReducer>> gpus_;
    std::atomic<std::size_t> next_{0};
    std::atomic<double> scheduler_accum_us_{0.0};
    std::vector<double> throughput_;
    std::mutex throughput_mutex_;
};

}  // namespace

std::unique_ptr<IReductionStrategy> make_strategy(const WorkerConfig& config, Dataset& dataset) {
    if (config.cpu_backend != CpuBackendKind::None && !cpu_backend_supported(config.cpu_backend)) {
        throw std::runtime_error("requested CPU backend is unavailable in this build");
    }
    const bool needs_gpu = config.scheduler != SchedulerKind::CpuOnly;
    if (needs_gpu && !gpu_runtime_available()) {
        throw std::runtime_error(gpu_unavailable_reason());
    }
    switch (config.scheduler) {
        case SchedulerKind::CpuOnly:
            return std::make_unique<CpuOnlyStrategy>(config, dataset);
        case SchedulerKind::GpuOnly:
            return std::make_unique<GpuOnlyStrategy>(config, dataset);
        case SchedulerKind::GpuStaticEqual:
            return std::make_unique<MultiGpuStrategy>(config, dataset, false);
        case SchedulerKind::GpuStaticProfiled:
            return std::make_unique<MultiGpuStrategy>(config, dataset, true);
        case SchedulerKind::StaticEqual:
            return std::make_unique<StaticHybridStrategy>(config, dataset, false);
        case SchedulerKind::StaticProfiled:
            return std::make_unique<StaticHybridStrategy>(config, dataset, true);
        case SchedulerKind::DynamicFixed:
        case SchedulerKind::DynamicGuided:
        case SchedulerKind::DynamicAdaptive:
            return std::make_unique<DynamicHybridStrategy>(config, dataset, config.scheduler);
    }
    throw std::logic_error("unreachable scheduler");
}

}  // namespace prbench
