#include "prbench/dataset.hpp"

#include <fstream>
#include <stdexcept>

namespace prbench {

Dataset::Dataset(const std::filesystem::path& path, std::size_t count, DataType dtype)
    : count_(count), dtype_(dtype) {
    const std::size_t expected = count * data_type_size(dtype);
    const auto actual = std::filesystem::file_size(path);
    if (actual != expected) {
        throw std::runtime_error(
            "dataset size mismatch: expected " + std::to_string(expected) +
            " bytes, got " + std::to_string(actual)
        );
    }
    bytes_.resize(expected);
    std::ifstream in(path, std::ios::binary);
    if (!in) throw std::runtime_error("failed to open dataset: " + path.string());
    in.read(reinterpret_cast<char*>(bytes_.data()), static_cast<std::streamsize>(bytes_.size()));
    if (!in) throw std::runtime_error("failed to read complete dataset: " + path.string());
}

}  // namespace prbench
