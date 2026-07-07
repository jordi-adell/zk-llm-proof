from pathlib import Path

import ezkl

from zkllms import backend

_CALIBRATION_TARGET = "resources"


def _private_run_args() -> "ezkl.PyRunArgs":
    run_args = ezkl.PyRunArgs()
    run_args.param_visibility = "private"
    run_args.input_visibility = "hashed"
    run_args.output_visibility = "public"
    return run_args


def compile_model(
    onnx_path: Path,
    input_json: Path,
    settings_path: Path,
    circuit_path: Path,
) -> None:
    backend.run(ezkl.gen_settings, str(onnx_path), str(settings_path), _private_run_args())
    backend.run(
        ezkl.calibrate_settings,
        str(input_json),
        str(onnx_path),
        str(settings_path),
        _CALIBRATION_TARGET,
    )
    backend.run(ezkl.compile_circuit, str(onnx_path), str(circuit_path), str(settings_path))


def setup_keys(
    circuit_path: Path,
    srs_path: Path,
    pk_path: Path,
    vk_path: Path,
    settings_path: Path,
) -> None:
    backend.run(ezkl.get_srs, str(settings_path), srs_path=str(srs_path))
    backend.run(ezkl.setup, str(circuit_path), str(vk_path), str(pk_path), str(srs_path))
