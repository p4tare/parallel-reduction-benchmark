#pragma once

#include <cstddef>
#include <cstdint>
#include <filesystem>
#include <vector>

#include "prbench/types.hpp"

namespace prbench {

class Dataset {
public:
    Dataset(const std::filesystem::path& path, std::size_t count, DataType dtype);

    const void* data() const noexcept { return bytes_.data(); }
    void* data() noexcept { return bytes_.data(); }
    std::size_t count() const noexcept { return count_; }
    DataType dtype() const noexcept { return dtype_; }
    std::size_t size_bytes() const noexcept { return bytes_.size(); }

    const void* offset_ptr(std::size_t element_offset) const noexcept {
        return bytes_.data() + element_offset * data_type_size(dtype_);
    }
    void* offset_ptr(std::size_t element_offset) noexcept {
        return bytes_.data() + element_offset * data_type_size(dtype_);
    }

private:
    std::vector<std::byte> bytes_;
    std::size_t count_;
    DataType dtype_;
};

}  // namespace prbench
