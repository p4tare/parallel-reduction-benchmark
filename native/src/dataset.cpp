#include "prbench/dataset.hpp"

#include <algorithm>
#include <cstring>
#include <fstream>
#include <limits>
#include <stdexcept>

namespace prbench {

Dataset::Dataset(
    const std::filesystem::path& path,
    std::size_t count,
    DataType dtype,
    std::size_t cache_rotation_target_bytes,
    std::size_t cache_rotation_max_replicas
)
    : count_(count), dtype_(dtype) {
    replica_bytes_ = count * data_type_size(dtype);
    const auto actual = std::filesystem::file_size(path);
    if (actual != replica_bytes_) {
        throw std::runtime_error(
            "dataset size mismatch: expected " + std::to_string(replica_bytes_) +
            " bytes, got " + std::to_string(actual)
        );
    }

    if (cache_rotation_target_bytes > 0 && replica_bytes_ > 0) {
        const std::size_t desired = (cache_rotation_target_bytes + replica_bytes_ - 1) / replica_bytes_;
        replica_count_ = std::max<std::size_t>(
            1,
            std::min<std::size_t>(cache_rotation_max_replicas, desired)
        );
    }
    if (replica_bytes_ > 0 && replica_count_ > std::numeric_limits<std::size_t>::max() / replica_bytes_) {
        throw std::overflow_error("dataset replica allocation size overflow");
    }

    bytes_.resize(replica_bytes_ * replica_count_);
    std::ifstream in(path, std::ios::binary);
    if (!in) throw std::runtime_error("failed to open dataset: " + path.string());
    in.read(reinterpret_cast<char*>(bytes_.data()), static_cast<std::streamsize>(replica_bytes_));
    if (!in) throw std::runtime_error("failed to read complete dataset: " + path.string());

    // Replicas are byte-identical, so every iteration has the same mathematical input.
    // They are created before warm-up/timing and only the active pointer is rotated later.
    for (std::size_t i = 1; i < replica_count_; ++i) {
        std::memcpy(bytes_.data() + i * replica_bytes_, bytes_.data(), replica_bytes_);
    }
}

}  // namespace prbench
