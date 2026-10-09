from types import SimpleNamespace

import pytest
import torch

from revllm import profile


def test_run_successful_updates_retries_overflows_without_counting_them():
    outcomes = iter([False, True, False, True, True])

    attempted, overflows = profile._run_successful_updates(lambda: next(outcomes), 3)

    assert attempted == 5
    assert overflows == 2


def test_run_successful_updates_bounds_repeated_overflows():
    with pytest.raises(RuntimeError, match="Exceeded 2 AMP overflow retries"):
        profile._run_successful_updates(lambda: False, 1, max_overflow_retries=2)


def test_benchmark_step_accumulates_the_exact_effective_batch():
    class LossModel(torch.nn.Module):
        def __init__(self):
            super().__init__()
            self.linear = torch.nn.Linear(2, 2, bias=False)

        def forward(self, x, y):
            return (self.linear(x) - y).square().sum()

    torch.manual_seed(7)
    full = LossModel()
    split = LossModel()
    split.load_state_dict(full.state_dict())
    x = torch.randn(3, 2)
    y = torch.randn(3, 2)
    scaler = torch.amp.GradScaler("cuda", enabled=False)
    full_optimizer = torch.optim.SGD(full.parameters(), lr=0.01)
    split_optimizer = torch.optim.SGD(split.parameters(), lr=0.01)

    assert profile._one_step(full, full_optimizer, scaler, x, y, False, physical_batch=3)
    assert profile._one_step(split, split_optimizer, scaler, x, y, False, physical_batch=2)
    torch.testing.assert_close(full.linear.weight, split.linear.weight, rtol=1e-6, atol=1e-7)


def test_cuda_metadata_records_runtime_device_memory_allocator_and_precision(monkeypatch):
    device = torch.device("cuda", 2)
    properties = SimpleNamespace(uuid="GPU-test-uuid")
    monkeypatch.setattr(torch.cuda, "mem_get_info", lambda selected: (3_000, 4_000))
    monkeypatch.setattr(torch.cuda, "get_device_capability", lambda selected: (8, 6))
    monkeypatch.setattr(torch.cuda, "get_device_properties", lambda selected: properties)
    monkeypatch.setattr(torch.cuda, "get_device_name", lambda selected: "Test GPU")
    monkeypatch.setattr(torch.cuda.memory, "get_allocator_backend", lambda: "cudaMallocAsync")
    monkeypatch.setattr(torch.backends.cuda.matmul, "allow_tf32", True)
    monkeypatch.setattr(torch.backends.cudnn, "allow_tf32", False)
    monkeypatch.setattr(torch.backends.cudnn, "version", lambda: 9999)

    result = profile._cuda_metadata(device, amp=True)

    assert result["device"] == "cuda:2"
    assert result["device_index"] == 2
    assert result["gpu_name"] == "Test GPU"
    assert result["gpu_uuid"] == "GPU-test-uuid"
    assert result["compute_capability"] == [8, 6]
    assert result["free_memory_bytes"] == 3_000
    assert result["total_memory_bytes"] == 4_000
    assert result["allocator_backend"] == "cudaMallocAsync"
    assert result["precision"] == "mixed_float16"
    assert result["autocast_dtype"] == "float16"
    assert result["amp_fp16"] is True
    assert result["backend"] == "cuda"
    assert result["device_backend"] == "cuda"
    assert result["pytorch_version"] == torch.__version__
    assert result["cudnn_version"] == 9999
    assert result["tf32_matmul_allowed"] is True
    assert result["tf32_cudnn_allowed"] is False


def test_elapsed_throughput_uses_actual_measured_token_count(monkeypatch):
    monkeypatch.setattr(profile.time, "perf_counter", lambda: 12.5)

    seconds, throughput = profile._elapsed_and_throughput(10.0, measured_tokens=1_000)

    assert seconds == 2.5
    assert throughput == 400.0


def test_benchmark_defaults_to_three_20_warmup_100_measured_repetitions(monkeypatch):
    cfg = SimpleNamespace(
        use_amp=True,
        attention_backend="auto",
        physical_batch=4,
        effective_batch=16,
        model=SimpleNamespace(block_size=512),
    )
    calls = []
    monkeypatch.setattr(profile, "_cuda_device", lambda: torch.device("cuda", 0))
    monkeypatch.setattr(profile, "prepare_attention_backend", lambda *args, **kwargs: {"effective_policy": "auto"})
    monkeypatch.setattr(profile, "_cuda_metadata", lambda device, amp: {"device": str(device)})
    monkeypatch.setattr(torch.cuda, "synchronize", lambda device: None)
    monkeypatch.setattr(torch.cuda, "empty_cache", lambda: None)

    def fake_repetition(config, device, repetition, warmup, measured):
        calls.append((repetition, warmup, measured))
        return {
            "repetition": repetition,
            "tokens_per_second": 100.0 + repetition,
            "measured_tokens": 204_800,
            "attempted_updates": 100,
            "overflow_retries": 0,
        }

    monkeypatch.setattr(profile, "_benchmark_repetition", fake_repetition)

    result = profile.benchmark(cfg)

    assert calls == [(0, 20, 100), (1, 20, 100), (2, 20, 100)]
    assert result["repetition_count"] == 3
    assert result["median_tokens_per_second"] == 101.0
    assert result["total_measured_tokens"] == 614_400
    assert result["total_attempted_updates"] == 300
    assert result["total_overflow_retries"] == 0
    assert result["physical_batch"] == 4
    assert result["effective_batch"] == 16
    assert result["microbatches_per_update"] == 4


@pytest.mark.parametrize("elapsed", [0.0, -1.0, float("inf"), float("nan")])
def test_elapsed_throughput_rejects_invalid_timing(monkeypatch, elapsed):
    monkeypatch.setattr(profile.time, "perf_counter", lambda: elapsed)
    with pytest.raises(RuntimeError, match="Invalid elapsed benchmark time"):
        profile._elapsed_and_throughput(0.0, measured_tokens=100)
