import time
from contextlib import contextmanager
from dataclasses import dataclass

import psutil

_BYTES_PER_MB = 1024 * 1024


def estimate_transformer_flops(
    d_model: int, n_heads: int, seq_len: int, n_layers: int, intermediate: int
) -> int:
    attention = 4 * seq_len * d_model * d_model + 2 * seq_len * seq_len * d_model
    feed_forward = 4 * seq_len * d_model * intermediate
    return (attention + feed_forward) * n_layers


@dataclass
class PhaseMetrics:
    name: str
    wall_s: float
    cpu_s: float
    cpu_util_pct: float
    peak_rss_mb: float


@dataclass
class TEEProfile:
    phases: list[PhaseMetrics]
    proof_size_bytes: int
    constraint_count: int
    inference_flops: int

    def summary_table(self) -> str:
        header = f"{'Phase':<14}{'Wall (s)':>10}{'CPU (s)':>10}{'CPU util%':>11}{'Peak RAM (MB)':>15}"
        rule = "-" * len(header)
        rows = [
            f"{p.name:<14}{p.wall_s:>10.3f}{p.cpu_s:>10.3f}{p.cpu_util_pct:>10.1f}%{p.peak_rss_mb:>15.2f}"
            for p in self.phases
        ]
        footer = [
            f"Circuit constraints: {self.constraint_count:>12,}",
            f"Proof size:          {self.proof_size_bytes / 1024:>9.1f} KB",
            f"Inference FLOPs:     {self.inference_flops:>12,}",
        ]
        return "\n".join(["TEE Firmware Computational Cost", rule, header, rule, *rows, rule, *footer])


@contextmanager
def tee_phase(name: str, phases: list[PhaseMetrics]):
    process = psutil.Process()
    rss_before = process.memory_info().rss
    wall_start = time.perf_counter()
    cpu_start = time.process_time()
    try:
        yield
    finally:
        wall_s = time.perf_counter() - wall_start
        cpu_s = time.process_time() - cpu_start
        rss_delta = process.memory_info().rss - rss_before
        cpu_util_pct = (cpu_s / wall_s * 100.0) if wall_s > 0 else 0.0
        phases.append(
            PhaseMetrics(
                name=name,
                wall_s=wall_s,
                cpu_s=cpu_s,
                cpu_util_pct=cpu_util_pct,
                peak_rss_mb=max(rss_delta, 0) / _BYTES_PER_MB,
            )
        )
