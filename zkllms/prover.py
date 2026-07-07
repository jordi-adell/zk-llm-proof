import json
from dataclasses import dataclass
from pathlib import Path

import ezkl

from zkllms import backend
from zkllms.instrumentation import PhaseMetrics, TEEProfile, tee_phase


@dataclass
class ProofResult:
    proof_path: Path
    output_values: list[float]
    profile: TEEProfile | None


def _extract_output_values(witness_path: Path) -> list[float]:
    data = json.loads(Path(witness_path).read_text())
    rescaled = data.get("pretty_elements", {}).get("rescaled_outputs") or []
    flat = rescaled[0] if rescaled else []
    return [float(value) for value in flat]


def _read_constraint_count(settings_path: Path) -> int:
    data = json.loads(Path(settings_path).read_text())
    return int(data.get("num_rows", 0))


def generate_proof(
    circuit_path: Path,
    input_json: Path,
    pk_path: Path,
    srs_path: Path,
    witness_path: Path,
    proof_path: Path,
    settings_path: Path,
    profile: bool = False,
    phases: list[PhaseMetrics] | None = None,
    inference_flops: int = 0,
) -> ProofResult:
    phases = phases if phases is not None else []
    with tee_phase("gen_witness", phases):
        backend.run(ezkl.gen_witness, str(input_json), str(circuit_path), str(witness_path))
    with tee_phase("prove", phases):
        backend.run(
            ezkl.prove,
            str(witness_path),
            str(circuit_path),
            str(pk_path),
            str(proof_path),
            str(srs_path),
        )

    output_values = _extract_output_values(witness_path)
    tee_profile = None
    if profile:
        tee_profile = TEEProfile(
            phases=phases,
            proof_size_bytes=Path(proof_path).stat().st_size,
            constraint_count=_read_constraint_count(settings_path),
            inference_flops=inference_flops,
        )
    return ProofResult(Path(proof_path), output_values, tee_profile)
