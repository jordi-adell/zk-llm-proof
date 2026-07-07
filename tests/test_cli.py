from click.testing import CliRunner

from zkllms.cli import cli
from zkllms.prover import ProofResult


def test_verify_reports_success_for_a_valid_proof(mocker, tmp_path):
    mocker.patch("zkllms.cli.verifier.verify_proof", return_value=True)
    proof = tmp_path / "proof.json"
    proof.write_text("{}")

    result = CliRunner().invoke(
        cli, ["verify", "--keys-dir", str(tmp_path), "--proof", str(proof)]
    )

    assert result.exit_code == 0
    assert "verified" in result.output.lower()


def test_verify_fails_nonzero_for_an_invalid_proof(mocker, tmp_path):
    mocker.patch("zkllms.cli.verifier.verify_proof", return_value=False)
    proof = tmp_path / "proof.json"
    proof.write_text("{}")

    result = CliRunner().invoke(
        cli, ["verify", "--keys-dir", str(tmp_path), "--proof", str(proof)]
    )

    assert result.exit_code != 0


def test_export_writes_onnx_and_calibration(mocker, tmp_path):
    mocker.patch("zkllms.cli.model.load_model", return_value=mocker.Mock())
    export_fn = mocker.patch("zkllms.cli.model.export_to_onnx")
    mocker.patch(
        "zkllms.cli.model.create_sample_input", return_value={"input_data": [[0.0]]}
    )
    onnx_path = tmp_path / "model.onnx"

    result = CliRunner().invoke(cli, ["export", "--output", str(onnx_path)])

    assert result.exit_code == 0
    export_fn.assert_called_once()
    assert (tmp_path / "model.onnx.calibration.json").exists()


def test_export_passes_the_model_name_to_load_model(mocker, tmp_path):
    load = mocker.patch("zkllms.cli.model.load_model", return_value=mocker.Mock())
    mocker.patch("zkllms.cli.model.export_to_onnx")
    mocker.patch("zkllms.cli.model.create_sample_input", return_value={"input_data": [[0.0]]})

    result = CliRunner().invoke(
        cli,
        [
            "export",
            "--output", str(tmp_path / "model.onnx"),
            "--model-name", "meta-llama/Llama-3.2-1B",
            "--num-layers", "2",
        ],
    )

    assert result.exit_code == 0
    assert load.call_args.kwargs["model_name"] == "meta-llama/Llama-3.2-1B"
    assert load.call_args.kwargs["num_layers"] == 2


def test_prove_passes_the_model_name_to_load_model(mocker, tmp_path):
    model_slice = mocker.Mock(d_model=8, n_heads=2, intermediate_size=16)
    load = mocker.patch("zkllms.cli.model.load_model", return_value=model_slice)
    mocker.patch("zkllms.cli.model.create_sample_input", return_value={"input_data": [[0.0]]})
    mocker.patch("zkllms.cli.model.project_hidden_to_tokens", return_value=([1], "x"))
    mocker.patch(
        "zkllms.cli.prover.generate_proof",
        return_value=ProofResult(tmp_path / "proof.json", [0.1], None),
    )
    model_path = tmp_path / "model.onnx"
    model_path.write_text("onnx")

    result = CliRunner().invoke(
        cli,
        [
            "prove",
            "--model", str(model_path),
            "--keys-dir", str(tmp_path),
            "--input", "hello",
            "--output", str(tmp_path / "proof.json"),
            "--seq-len", "1",
            "--model-name", "mistralai/Mistral-7B-v0.1",
        ],
    )

    assert result.exit_code == 0, result.output
    assert load.call_args.kwargs["model_name"] == "mistralai/Mistral-7B-v0.1"


def test_setup_compiles_circuit_and_generates_keys(mocker, tmp_path):
    compile_fn = mocker.patch("zkllms.cli.circuit.compile_model")
    setup_fn = mocker.patch("zkllms.cli.circuit.setup_keys")
    model_path = tmp_path / "model.onnx"
    model_path.write_text("onnx")
    keys_dir = tmp_path / "keys"

    result = CliRunner().invoke(
        cli, ["setup", "--model", str(model_path), "--keys-dir", str(keys_dir)]
    )

    assert result.exit_code == 0
    compile_fn.assert_called_once()
    setup_fn.assert_called_once()


def test_prove_prints_inference_result_and_proof_location(mocker, tmp_path):
    mocker.patch(
        "zkllms.cli.model.load_model",
        return_value=mocker.Mock(d_model=8, n_heads=2, intermediate_size=16),
    )
    mocker.patch(
        "zkllms.cli.model.create_sample_input", return_value={"input_data": [[0.0, 0.0]]}
    )
    mocker.patch(
        "zkllms.cli.model.project_hidden_to_tokens",
        return_value=([5, 6, 7], "hi there"),
    )
    mocker.patch(
        "zkllms.cli.prover.generate_proof",
        return_value=ProofResult(
            proof_path=tmp_path / "proof.json",
            output_values=[0.1, 0.9, 0.2, 0.8, 0.3, 0.7],
            profile=None,
        ),
    )
    model_path = tmp_path / "model.onnx"
    model_path.write_text("onnx")

    result = CliRunner().invoke(
        cli,
        [
            "prove",
            "--model", str(model_path),
            "--keys-dir", str(tmp_path),
            "--input", "hello",
            "--output", str(tmp_path / "proof.json"),
            "--seq-len", "3",
        ],
    )

    assert result.exit_code == 0, result.output
    assert "Inference result" in result.output
    assert "hi there" in result.output
    assert "Proof written to" in result.output
