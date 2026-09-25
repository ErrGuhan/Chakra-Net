"""Unit tests for ChakraNet Evaluation Metrics Module (Milestone 3)"""

import pytest
import numpy as np
from models.metrics import (
    calculate_radially_averaged_psd,
    calculate_psd_retention,
    calculate_p99_bias,
    calculate_crps_ensemble,
    benchmark_inference_latency,
    generate_milestone3_metrics,
)


def test_calculate_radially_averaged_psd():
    field = np.random.randn(64, 64).astype(np.float32)
    k, psd = calculate_radially_averaged_psd(field)

    assert len(k) == 32
    assert len(psd) == 32
    assert np.all(psd >= 0.0)


def test_calculate_psd_retention():
    target = np.random.randn(64, 64).astype(np.float32) * 50.0
    pred = target.copy()  # Perfect reconstruction

    retention = calculate_psd_retention(pred, target, nyquist_cutoff_fraction=0.3)
    assert pytest.approx(retention, rel=1e-2) == 100.0


def test_calculate_p99_bias():
    target = np.ones((64, 64), dtype=np.float32) * 100.0
    pred = np.ones((64, 64), dtype=np.float32) * 105.0

    bias = calculate_p99_bias(pred, target)
    assert pytest.approx(bias, abs=1e-2) == 5.0


def test_calculate_crps_ensemble():
    # 8 realizations of 32x32
    ens = np.random.uniform(10, 20, (8, 32, 32)).astype(np.float32)
    target = np.ones((32, 32), dtype=np.float32) * 15.0

    crps_val = calculate_crps_ensemble(ens, target)
    assert crps_val >= 0.0
    assert crps_val < 10.0


def test_benchmark_inference_latency():
    def dummy_func(x):
        return np.mean(x)

    sample = np.ones((10, 10))
    res = benchmark_inference_latency(dummy_func, sample, warmup=1, runs=3)
    assert "mean_latency_ms" in res
    assert res["mean_latency_ms"] >= 0.0


def test_generate_milestone3_metrics_integration(tmp_path):
    json_path = str(tmp_path / "metrics_test.json")
    md_path = str(tmp_path / "table_test.md")

    res = generate_milestone3_metrics(output_json_path=json_path, output_table_path=md_path)
    assert "kpis" in res
    assert "psd_high_frequency_retention" in res["kpis"]
    assert "p99_rainfall_bias" in res["kpis"]
    assert "crps_improvement_over_bilinear" in res["kpis"]
