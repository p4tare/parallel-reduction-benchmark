#pragma once

#include <memory>
#include <string>

#include "prbench/cli.hpp"
#include "prbench/strategy.hpp"

namespace prbench {

// New memory-path and out-of-core algorithms use the same IReductionStrategy
// protocol as the original benchmark. Legacy algorithms stay in strategy.cpp.
bool uses_integrated_strategy(const WorkerConfig& config) noexcept;
std::string integrated_unsupported_reason(const WorkerConfig& config);
std::unique_ptr<IReductionStrategy> make_integrated_strategy(const WorkerConfig& config);

}  // namespace prbench
