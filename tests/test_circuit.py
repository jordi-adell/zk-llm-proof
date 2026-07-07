from zkllms import circuit


def test_compile_model_hides_weights_and_runs_the_pipeline(mocker, tmp_path):
    gen = mocker.patch("zkllms.circuit.ezkl.gen_settings")
    cal = mocker.patch("zkllms.circuit.ezkl.calibrate_settings")
    comp = mocker.patch("zkllms.circuit.ezkl.compile_circuit")

    circuit.compile_model(
        onnx_path=tmp_path / "model.onnx",
        input_json=tmp_path / "input.json",
        settings_path=tmp_path / "settings.json",
        circuit_path=tmp_path / "model.compiled",
    )

    run_args = gen.call_args.args[2] if len(gen.call_args.args) > 2 else gen.call_args.kwargs["py_run_args"]
    assert run_args.param_visibility == "private"
    assert run_args.input_visibility.startswith("hashed")
    assert run_args.output_visibility == "public"
    gen.assert_called_once()
    cal.assert_called_once()
    comp.assert_called_once()


def test_compile_model_runs_pipeline_in_dependency_order(mocker, tmp_path):
    manager = mocker.Mock()
    manager.attach_mock(mocker.patch("zkllms.circuit.ezkl.gen_settings"), "gen")
    manager.attach_mock(mocker.patch("zkllms.circuit.ezkl.calibrate_settings"), "cal")
    manager.attach_mock(mocker.patch("zkllms.circuit.ezkl.compile_circuit"), "comp")

    circuit.compile_model(
        onnx_path=tmp_path / "model.onnx",
        input_json=tmp_path / "input.json",
        settings_path=tmp_path / "settings.json",
        circuit_path=tmp_path / "model.compiled",
    )

    assert [c[0] for c in manager.mock_calls] == ["gen", "cal", "comp"]


def test_setup_keys_downloads_srs_then_generates_keys(mocker, tmp_path):
    manager = mocker.Mock()
    manager.attach_mock(mocker.patch("zkllms.circuit.ezkl.get_srs"), "get_srs")
    manager.attach_mock(mocker.patch("zkllms.circuit.ezkl.setup"), "setup")

    circuit.setup_keys(
        circuit_path=tmp_path / "model.compiled",
        srs_path=tmp_path / "kzg.srs",
        pk_path=tmp_path / "pk.key",
        vk_path=tmp_path / "vk.key",
        settings_path=tmp_path / "settings.json",
    )

    assert [c[0] for c in manager.mock_calls] == ["get_srs", "setup"]
