#pragma once

#include <cstddef>
#include <cstdint>
#include <filesystem>
#include <vector>

#include "prbench/types.hpp"

namespace prbench {

class Dataset {
public:
    Dataset(
        const std::filesystem::path& path,
        std::size_t count,
        DataType dtype,
        std::size_t cache_rotation_target_bytes,
        std::size_t cache_rotation_max_replicas
    );

    const void* data() const noexcept { return bytes_.data() + replica_index_ * replica_bytes_; }
    void* data() noexcept { return bytes_.data() + replica_index_ * replica_bytes_; }
    std::size_t count() const noexcept { return count_; }
    DataType dtype() const noexcept { return dtype_; }
    std::size_t size_bytes() const noexcept { return replica_bytes_; }
    std::size_t resident_bytes() const noexcept { return bytes_.size(); }
    std::size_t replica_count() const noexcept { return replica_count_; }
    std::size_t replica_index() const noexcept { return replica_index_; }
    void advance_replica() noexcept {
        if (replica_count_ > 1) replica_index_ = (replica_index_ + 1) % replica_count_;
    }

    const void* offset_ptr(std::size_t element_offset) const noexcept {
        return static_cast<const std::byte*>(data()) + element_offset * data_type_size(dtype_);
    }
    void* offset_ptr(std::size_t element_offset) noexcept {
        return static_cast<std::byte*>(data()) + element_offset * data_type_size(dtype_);
    }

private:
    std::vector<std::byte> bytes_;
    std::size_t count_;
    DataType dtype_;
    std::size_t replica_bytes_{0};
    std::size_t replica_count_{1};
    std::size_t replica_index_{0};
};

}  // namespace prbench
