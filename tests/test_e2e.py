import json

import pytest

from zkllms import circuit, prover, verifier


def _export_tiny_onnx(path):
    import torch

    class TinyNet(torch.nn.Module):
        def __init__(self):
            super().__init__()
            self.fc1 = torch.nn.Linear(4, 8)
            self.fc2 = torch.nn.Linear(8, 3)

        def forward(self, x):
            return self.fc2(torch.relu(self.fc1(x)))

    torch.onnx.export(
        TinyNet().eval(),
        (torch.randn(1, 4),),
        str(path),
        input_names=["x"],
        output_names=["y"],
        opset_version=17,
        dynamo=False,
    )


@pytest.mark.slow
def test_ezkl_pipeline_produces_a_verifiable_proof(tmp_path):
    onnx_path = tmp_path / "model.onnx"
    _export_tiny_onnx(onnx_path)
    sample = {"input_data": [[0.1, 0.2, 0.3, 0.4]]}
    calibration = tmp_path / "calibration.json"
    calibration.write_text(json.dumps(sample))

    settings = tmp_path / "settings.json"
    compiled = tmp_path / "model.compiled"
    srs, pk, vk = tmp_path / "kzg.srs", tmp_path / "pk.key", tmp_path / "vk.key"

    circuit.compile_model(onnx_path, calibration, settings, compiled)
    circuit.setup_keys(compiled, srs, pk, vk, settings)

    input_json = tmp_path / "input.json"
    input_json.write_text(json.dumps(sample))
    proof, witness = tmp_path / "proof.json", tmp_path / "witness.json"

    result = prover.generate_proof(
        compiled, input_json, pk, srs, witness, proof, settings, profile=True
    )

    assert proof.exists()
    assert len(result.output_values) == 3
    assert result.profile.proof_size_bytes > 0
    assert verifier.verify_proof(proof, vk, settings, srs) is True
