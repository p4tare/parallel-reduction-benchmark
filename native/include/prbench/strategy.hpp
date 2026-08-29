#pragma once

#include <memory>
#include <vector>

#include "prbench/cli.hpp"
#include "prbench/dataset.hpp"
#include "prbench/metrics.hpp"

namespace prbench {

// Return the dataset size used to calibrate the throughput-aware dynamic scheduler.
// Fixed/guided schedulers intentionally return zero because they do not consume
// throughput estimates.  Keeping this policy explicit makes reducer capacity
// planning testable independently from CUDA hardware.
std::size_t dynamic_calibration_elements(
    SchedulerKind kind,
    std::size_t dataset_count,
    std::size_t min_chunk_size,
    std::size_t max_chunk_size
);

class IReductionStrategy {
public:
    virtual ~IReductionStrategy() = default;
    virtual void prepare(int warmup_runs) = 0;
    virtual double warmup_median_us() const noexcept = 0;
    virtual const PrepareMetrics& prepare_metrics() const noexcept = 0;
    virtual IterationMetrics run_once() = 0;
};

std::unique_ptr<IReductionStrategy> make_strategy(const WorkerConfig& config, Dataset& dataset);

}  // namespace prbench
