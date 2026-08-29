from pathlib import Path


def test_cuda_check_macro_accepts_template_commas() -> None:
    source = (Path(__file__).resolve().parents[1] / "native" / "src" / "gpu_backend.cu").read_text(
        encoding="utf-8"
    )
    assert "#define CUDA_CHECK(...)" in source
    assert "CUDA_CHECK(cub_reduce<T, Op>(" in source


def test_cuda_metrics_report_processed_elements_and_library_versions() -> None:
    root = Path(__file__).resolve().parents[1]
    source = (root / "native" / "src" / "gpu_backend.cu").read_text(encoding="utf-8")
    header = (root / "native" / "include" / "prbench" / "gpu_backend.hpp").read_text(encoding="utf-8")
    assert "metrics.elements = count" in source
    assert "gpu_library_versions_json" in source
    assert "CUB_VERSION" in source
    assert "CCCL_VERSION" in source
    assert "gpu_library_versions_json" in header


def test_cuda_reduction_identity_avoids_host_only_numeric_limits_calls() -> None:
    source = (Path(__file__).resolve().parents[1] / "native" / "src" / "gpu_backend.cu").read_text(
        encoding="utf-8"
    )
    identity_region = source[source.index("reduction_max_finite"):source.index("reduction_combine")]
    assert "return std::numeric_limits" not in identity_region
    assert "0x1.fffffep+127f" in identity_region
    assert "0x1.fffffffffffffp+1023" in identity_region
