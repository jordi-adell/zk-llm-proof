import json

from zkllms import prover


def _write_witness(path, outputs):
    path.write_text(json.dumps({"pretty_elements": {"rescaled_outputs": outputs}}))


def test_generate_proof_runs_witness_then_prove(mocker, tmp_path):
    manager = mocker.Mock()
    manager.attach_mock(mocker.patch("zkllms.prover.ezkl.gen_witness"), "gen_witness")
    manager.attach_mock(mocker.patch("zkllms.prover.ezkl.prove"), "prove")
    witness = tmp_path / "witness.json"
    _write_witness(witness, [["0.1", "0.9", "0.2"]])

    result = prover.generate_proof(
        circuit_path=tmp_path / "model.compiled",
        input_json=tmp_path / "input.json",
        pk_path=tmp_path / "pk.key",
        srs_path=tmp_path / "kzg.srs",
        witness_path=witness,
        proof_path=tmp_path / "proof.json",
        settings_path=tmp_path / "settings.json",
    )

    assert [c[0] for c in manager.mock_calls] == ["gen_witness", "prove"]
    assert result.proof_path == tmp_path / "proof.json"
    assert result.output_values == [0.1, 0.9, 0.2]


def test_generate_proof_without_profile_has_no_profile(mocker, tmp_path):
    mocker.patch("zkllms.prover.ezkl.gen_witness")
    mocker.patch("zkllms.prover.ezkl.prove")
    witness = tmp_path / "witness.json"
    _write_witness(witness, [[1.0]])

    result = prover.generate_proof(
        circuit_path=tmp_path / "model.compiled",
        input_json=tmp_path / "input.json",
        pk_path=tmp_path / "pk.key",
        srs_path=tmp_path / "kzg.srs",
        witness_path=witness,
        proof_path=tmp_path / "proof.json",
        settings_path=tmp_path / "settings.json",
    )

    assert result.profile is None


def test_generate_proof_with_profile_reads_constraints_and_keeps_prior_phases(mocker, tmp_path):
    mocker.patch("zkllms.prover.ezkl.gen_witness")
    mocker.patch("zkllms.prover.ezkl.prove")
    witness = tmp_path / "witness.json"
    _write_witness(witness, [[1.0]])
    proof = tmp_path / "proof.json"
    proof.write_bytes(b"x" * 7065)
    settings = tmp_path / "settings.json"
    settings.write_text(json.dumps({"num_rows": 3_145_728}))
    prior = [prover.PhaseMetrics("decrypt", 0.001, 0.001, 100.0, 0.0)]

    result = prover.generate_proof(
        circuit_path=tmp_path / "model.compiled",
        input_json=tmp_path / "input.json",
        pk_path=tmp_path / "pk.key",
        srs_path=tmp_path / "kzg.srs",
        witness_path=witness,
        proof_path=proof,
        settings_path=settings,
        profile=True,
        phases=prior,
        inference_flops=186_400_000,
    )

    assert result.profile.constraint_count == 3_145_728
    assert result.profile.proof_size_bytes == 7065
    assert result.profile.inference_flops == 186_400_000
    assert [p.name for p in result.profile.phases] == ["decrypt", "gen_witness", "prove"]
