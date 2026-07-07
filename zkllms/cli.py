import json
from pathlib import Path

import click

from zkllms import circuit, crypto, model, prover, verifier
from zkllms.instrumentation import estimate_transformer_flops, tee_phase

_KDF_SALT = b"zkllms-demo-salt"


def _key_paths(keys_dir: Path) -> dict[str, Path]:
    keys_dir = Path(keys_dir)
    return {
        "settings": keys_dir / "settings.json",
        "circuit": keys_dir / "model.compiled",
        "srs": keys_dir / "kzg.srs",
        "pk": keys_dir / "pk.key",
        "vk": keys_dir / "vk.key",
    }


def _calibration_path(onnx_path: Path) -> Path:
    return Path(str(onnx_path) + ".calibration.json")


@click.group()
def cli() -> None:
    pass


@cli.command()
@click.option("--output", type=click.Path(), default="model.onnx")
@click.option("--model-name", default=model.DEFAULT_MODEL_NAME, show_default=True)
@click.option("--seq-len", default=4, show_default=True)
@click.option("--num-layers", default=1, show_default=True)
def export(output: str, model_name: str, seq_len: int, num_layers: int) -> None:
    model_slice = model.load_model(model_name=model_name, num_layers=num_layers)
    onnx_path = Path(output)
    model.export_to_onnx(model_slice, onnx_path, seq_len=seq_len)
    calibration = model.create_sample_input(model_slice, "hello world", seq_len=seq_len)
    _calibration_path(onnx_path).write_text(json.dumps(calibration))
    click.echo(f"Exported {num_layers}-layer model to {onnx_path}")


@cli.command()
@click.option("--model", "model_path", type=click.Path(exists=True), required=True)
@click.option("--keys-dir", type=click.Path(), required=True)
def setup(model_path: str, keys_dir: str) -> None:
    Path(keys_dir).mkdir(parents=True, exist_ok=True)
    keys = _key_paths(keys_dir)
    circuit.compile_model(
        Path(model_path), _calibration_path(model_path), keys["settings"], keys["circuit"]
    )
    circuit.setup_keys(keys["circuit"], keys["srs"], keys["pk"], keys["vk"], keys["settings"])
    click.echo(f"Proving/verification keys written to {keys_dir}")


@cli.command()
@click.option("--model", "model_path", type=click.Path(exists=True), required=True)
@click.option("--keys-dir", type=click.Path(exists=True), required=True)
@click.option("--input", "text", required=True)
@click.option("--output", "proof_out", type=click.Path(), default="proof.json")
@click.option("--model-name", default=model.DEFAULT_MODEL_NAME, show_default=True)
@click.option("--seq-len", default=4, show_default=True)
@click.option("--num-layers", default=1, show_default=True)
@click.option("--passphrase", default=None)
@click.option("--profile/--no-profile", default=True)
def prove(
    model_path: str,
    keys_dir: str,
    text: str,
    proof_out: str,
    model_name: str,
    seq_len: int,
    num_layers: int,
    passphrase: str | None,
    profile: bool,
) -> None:
    keys = _key_paths(keys_dir)
    phases = []
    key = crypto.derive_key(passphrase, _KDF_SALT) if passphrase else crypto.generate_key()
    ciphertext, nonce = crypto.encrypt_weights(Path(model_path), key)
    with tee_phase("decrypt", phases):
        crypto.decrypt_bytes(ciphertext, nonce, key)

    model_slice = model.load_model(model_name=model_name, num_layers=num_layers)
    sample = model.create_sample_input(model_slice, text, seq_len=seq_len)
    input_json = Path(keys_dir) / "input.json"
    input_json.write_text(json.dumps(sample))
    witness = Path(keys_dir) / "witness.json"
    flops = estimate_transformer_flops(
        model_slice.d_model,
        model_slice.n_heads,
        seq_len,
        num_layers,
        model_slice.intermediate_size,
    )

    result = prover.generate_proof(
        circuit_path=keys["circuit"],
        input_json=input_json,
        pk_path=keys["pk"],
        srs_path=keys["srs"],
        witness_path=witness,
        proof_path=Path(proof_out),
        settings_path=keys["settings"],
        profile=profile,
        phases=phases,
        inference_flops=flops,
    )

    tokens, decoded = model.project_hidden_to_tokens(model_slice, result.output_values, seq_len)
    click.echo("Inference result")
    click.echo(f"Input:  {text}")
    click.echo(f"Output: {decoded}")
    click.echo(f"Tokens: {tokens}")
    click.echo(f"Proof written to {proof_out}")
    if result.profile is not None:
        click.echo(result.profile.summary_table())


@cli.command()
@click.option("--keys-dir", type=click.Path(exists=True), required=True)
@click.option("--proof", type=click.Path(exists=True), required=True)
def verify(keys_dir: str, proof: str) -> None:
    keys = _key_paths(keys_dir)
    accepted = verifier.verify_proof(Path(proof), keys["vk"], keys["settings"], keys["srs"])
    if not accepted:
        raise click.ClickException("Proof INVALID")
    click.echo("Proof verified ✓")
