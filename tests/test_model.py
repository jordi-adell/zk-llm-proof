import pytest

from zkllms import model


def test_decode_logits_to_tokens_argmaxes_each_position():
    values = [0.1, 0.9, 0.8, 0.2, 0.3, 0.7]

    tokens = model.decode_logits_to_tokens(values, seq_len=3, vocab_size=2)

    assert tokens == [1, 0, 1]


def test_decode_logits_to_tokens_rejects_mismatched_length():
    with pytest.raises(ValueError):
        model.decode_logits_to_tokens([0.1, 0.2, 0.3], seq_len=2, vocab_size=2)


@pytest.mark.slow
def test_create_sample_input_produces_flattened_hidden_states():
    model_slice = model.load_model(num_layers=1)

    sample = model.create_sample_input(model_slice, "Hello world", seq_len=4)

    assert list(sample.keys()) == ["input_data"]
    assert len(sample["input_data"][0]) == 4 * model_slice.d_model


@pytest.mark.slow
def test_export_to_onnx_drops_vocab_tables_and_stays_small(tmp_path):
    model_slice = model.load_model(num_layers=1)
    onnx_path = tmp_path / "model.onnx"

    model.export_to_onnx(model_slice, onnx_path, seq_len=4)

    import onnx as onnx_lib

    graph = onnx_lib.load(str(onnx_path))
    onnx_lib.checker.check_model(graph)
    assert onnx_path.stat().st_size < 100 * 1024 * 1024


@pytest.mark.slow
def test_load_model_supports_a_non_qwen_decoder_llm(tmp_path):
    model_slice = model.load_model(
        model_name="hf-internal-testing/tiny-random-LlamaForCausalLM", num_layers=1
    )

    sample = model.create_sample_input(model_slice, "hello world", seq_len=2)
    model.export_to_onnx(model_slice, tmp_path / "llama.onnx", seq_len=2)

    assert model_slice.d_model > 0
    assert len(sample["input_data"][0]) == 2 * model_slice.d_model
    assert (tmp_path / "llama.onnx").exists()


@pytest.mark.slow
def test_load_model_rejects_unsupported_architecture():
    with pytest.raises(ValueError, match="unsupported architecture"):
        model.load_model(model_name="hf-internal-testing/tiny-random-gpt2", num_layers=1)


@pytest.mark.slow
def test_load_model_caches_into_the_given_directory(tmp_path):
    cache = tmp_path / "models"

    model.load_model(
        model_name="hf-internal-testing/tiny-random-LlamaForCausalLM",
        num_layers=1,
        cache_dir=str(cache),
    )

    assert cache.exists()
    assert any(cache.rglob("*.safetensors")) or any(cache.rglob("*.bin"))
