from zkllms import verifier


def test_verify_proof_returns_the_ezkl_verdict(mocker, tmp_path):
    mocker.patch("zkllms.verifier.ezkl.verify", return_value=True)

    result = verifier.verify_proof(
        proof_path=tmp_path / "proof.json",
        vk_path=tmp_path / "vk.key",
        settings_path=tmp_path / "settings.json",
        srs_path=tmp_path / "kzg.srs",
    )

    assert result is True


def test_verify_proof_passes_paths_in_ezkl_argument_order(mocker, tmp_path):
    verify = mocker.patch("zkllms.verifier.ezkl.verify", return_value=True)

    verifier.verify_proof(
        proof_path=tmp_path / "proof.json",
        vk_path=tmp_path / "vk.key",
        settings_path=tmp_path / "settings.json",
        srs_path=tmp_path / "kzg.srs",
    )

    proof, settings, vk = verify.call_args.args[:3]
    assert proof.endswith("proof.json")
    assert settings.endswith("settings.json")
    assert vk.endswith("vk.key")
