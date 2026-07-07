from dataclasses import dataclass
from pathlib import Path

DEFAULT_MODEL_NAME = "Qwen/Qwen2.5-0.5B"


@dataclass
class ModelSlice:
    tokenizer: object
    zk_model: object
    embedder: object
    head: object
    d_model: int
    n_heads: int
    intermediate_size: int
    vocab_size: int


def _config_value(config, *names: str, default: int = 0) -> int:
    for name in names:
        value = getattr(config, name, None)
        if value is not None:
            return value
    return default


def decode_logits_to_tokens(values: list[float], seq_len: int, vocab_size: int) -> list[int]:
    if len(values) != seq_len * vocab_size:
        raise ValueError(f"expected {seq_len * vocab_size} logits, got {len(values)}")
    tokens = []
    for position in range(seq_len):
        row = values[position * vocab_size : (position + 1) * vocab_size]
        tokens.append(max(range(vocab_size), key=row.__getitem__))
    return tokens


def load_model(model_name: str = DEFAULT_MODEL_NAME, num_layers: int = 1) -> ModelSlice:
    import torch
    from transformers import AutoModelForCausalLM, AutoTokenizer

    tokenizer = AutoTokenizer.from_pretrained(model_name)
    if tokenizer.pad_token is None:
        tokenizer.pad_token = tokenizer.eos_token
    full = AutoModelForCausalLM.from_pretrained(
        model_name, dtype=torch.float32, attn_implementation="eager"
    )
    backbone = getattr(full, "model", None)
    if backbone is None or not hasattr(backbone, "layers") or not hasattr(backbone, "embed_tokens"):
        raise ValueError(
            f"{model_name}: unsupported architecture. zkllms proves the block of a "
            "rotary-based decoder causal LM (Llama, Qwen, Mistral, Gemma, Phi, ...) "
            "exposing model.layers and model.embed_tokens."
        )
    backbone.layers = backbone.layers[:num_layers]
    full.config.num_hidden_layers = num_layers
    embedder = backbone.embed_tokens
    head = full.get_output_embeddings()
    full.eval()

    class HiddenStateSlice(torch.nn.Module):
        def __init__(self, model):
            super().__init__()
            self.model = model

        def forward(self, inputs_embeds):
            seq_len = inputs_embeds.shape[1]
            position_ids = torch.arange(seq_len).unsqueeze(0)
            kwargs = {
                "attention_mask": torch.zeros(1, 1, seq_len, seq_len),
                "position_ids": position_ids,
            }
            rotary = getattr(self.model, "rotary_emb", None)
            if rotary is not None:
                kwargs["position_embeddings"] = rotary(inputs_embeds, position_ids)
            hidden = inputs_embeds
            for layer in self.model.layers:
                hidden = layer(hidden, **kwargs)
                if isinstance(hidden, tuple):
                    hidden = hidden[0]
            norm = getattr(self.model, "norm", None)
            return norm(hidden) if norm is not None else hidden

    config = full.config
    hidden_size = _config_value(config, "hidden_size", "n_embd", "d_model")
    return ModelSlice(
        tokenizer=tokenizer,
        zk_model=HiddenStateSlice(backbone),
        embedder=embedder,
        head=head,
        d_model=hidden_size,
        n_heads=_config_value(config, "num_attention_heads", "n_head"),
        intermediate_size=_config_value(
            config, "intermediate_size", "ffn_dim", "n_inner", default=4 * hidden_size
        ),
        vocab_size=_config_value(config, "vocab_size"),
    )


def _encode_ids(tokenizer, text: str, seq_len: int) -> list[int]:
    return tokenizer(
        text, padding="max_length", truncation=True, max_length=seq_len
    )["input_ids"]


def create_sample_input(model_slice: ModelSlice, text: str, seq_len: int = 4) -> dict:
    import torch

    ids = _encode_ids(model_slice.tokenizer, text, seq_len)
    with torch.no_grad():
        hidden = model_slice.embedder(torch.tensor([ids], dtype=torch.long))
    return {"input_data": [hidden.flatten().tolist()]}


def export_to_onnx(model_slice: ModelSlice, path: Path, seq_len: int = 4) -> None:
    import torch

    dummy = torch.zeros(1, seq_len, model_slice.d_model, dtype=torch.float32)
    torch.onnx.export(
        model_slice.zk_model,
        (dummy,),
        str(path),
        input_names=["inputs_embeds"],
        output_names=["hidden_states"],
        opset_version=17,
        dynamic_axes=None,
        dynamo=False,
    )


def project_hidden_to_tokens(
    model_slice: ModelSlice, output_values: list[float], seq_len: int
) -> tuple[list[int], str]:
    import torch

    hidden = torch.tensor(output_values, dtype=torch.float32).reshape(
        1, seq_len, model_slice.d_model
    )
    with torch.no_grad():
        logits = model_slice.head(hidden)
    tokens = decode_logits_to_tokens(
        logits.flatten().tolist(), seq_len, model_slice.vocab_size
    )
    return tokens, model_slice.tokenizer.decode(tokens)
