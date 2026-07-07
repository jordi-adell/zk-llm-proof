# zkllms

Zero-knowledge proof system for LLM inference on hardware accelerators.

## Overview

`zkllms` implements a zero-knowledge proof pipeline for transformer-based language
model inference, designed to run on hardware accelerator firmware. Both model weights
and input data are provided **encrypted** to the accelerator; the hardware executes
inference and emits a SNARK proof that the computation was performed correctly — without
ever revealing the weights or inputs to external observers.

The proof system uses [ezkl](https://github.com/zkonduit/ezkl) to compile an ONNX model
into a halo2 (PLONK/KZG) circuit. Correctness is enforced cryptographically by the ZK
layer; confidentiality of the executing computation is provided by a Trusted Execution
Environment (TEE) on the accelerator. This CLI prototype simulates the TEE / hardware
boundary in-process with AES-GCM encryption — the ZK math is real and sound.

The proven computation is the **transformer block** of a HuggingFace decoder LLM,
operating on hidden states: the token embedding and the final vocabulary projection
(`lm_head`) run on the host and are not part of the proof. The model is selected with
`--model-name` and defaults to [Qwen2.5-0.5B](https://huggingface.co/Qwen/Qwen2.5-0.5B);
any rotary-based decoder causal LM works (see [Supported models](#supported-models)). See
[Prototype scope](#prototype-scope) for why the scope is limited to the block.

## Goals

- **Privacy-preserving inference** — model weights and user data remain encrypted end-to-end
- **Hardware-targeted verifier** — the ZK verifier is designed to be implementable on
  accelerator firmware (GPU / FPGA / custom ASIC)
- **SNARK-based proofs** — halo2 / PLONK via `ezkl` for succinct, fast-to-verify proofs
- **CLI workflow** — export, setup, prove, and verify transformer inference from the command line

## Architecture

```
  Client (CLI)
  ┌─────────────────────────────────────────┐
  │ AES-GCM encrypt(weights, key) → c_w     │
  │ AES-GCM encrypt(input,   key) → c_x     │
  └─────────────────┬───────────────────────┘
                    │ (c_w, c_x)  [simulated secure channel]
  Hardware Accelerator [TEE boundary — simulated in-process]
  ┌─────────────────▼───────────────────────┐
  │ decrypt(c_w) → weights   (never leaves)  │
  │ decrypt(c_x) → input     (never leaves)  │
  │ ezkl gen_witness + prove → (output, π)   │
  └─────────────────┬───────────────────────┘
                    │ (output, π)  ← no plaintext leaves
  Verifier
  ┌─────────────────▼───────────────────────┐
  │ ezkl.verify(vk, π, output) → accept/reject│
  └─────────────────────────────────────────┘
```

ezkl visibility configuration realizes the privacy goal:

| Setting | Value | Effect |
|---------|-------|--------|
| `param_visibility` | `private` | weights hidden from the proof transcript |
| `input_visibility` | `hashed` | only a Poseidon hash of the input is public |
| `output_visibility` | `public` | output logits are public |

## Quickstart

```bash
pip install -e ".[dev]"

# 1. Export the first N layers of a model to ONNX (defaults to Qwen/Qwen2.5-0.5B).
#    The model is cached in ./models on first download.
zkllms export --output model.onnx --model-name Qwen/Qwen2.5-0.5B --models-dir models/ \
  --seq-len 4 --num-layers 1

# 2. Compile the model into a ZK circuit and generate proving/verification keys
zkllms setup --model model.onnx --keys-dir keys/

# 3. Run inference and generate a ZK proof (weights + input are encrypted in-process).
#    Pass the SAME --model-name and --models-dir used for export (reuses the cache).
zkllms prove --model model.onnx --keys-dir keys/ --model-name Qwen/Qwen2.5-0.5B \
  --models-dir models/ --input "Hello world" --output proof.json

# 4. Verify the proof
zkllms verify --keys-dir keys/ --proof proof.json
# Expected: "Proof verified ✓"
```

### Supported models

`--model-name` accepts any HuggingFace **rotary-based decoder causal LM** — Llama, Qwen,
Mistral, Gemma, Phi and similar architectures that expose `model.layers` and
`model.embed_tokens`. All shape parameters (hidden size, heads, FFN width, vocab) are read
from the model's `config`, so no per-model code changes are needed. Architectures outside
this family — encoder models (BERT), encoder-decoder models (T5), and legacy
absolute-position models (GPT-2) — are rejected with a clear error, because the
hidden-state block approach relies on the rotary decoder-layer structure.

### Model cache

`--models-dir` (default `models/`) is used as the HuggingFace download cache. Both
`export` and `prove` load the model, so caching means a model is downloaded once and
reused on every subsequent command — pass the same `--models-dir` to both. The directory
is created on first use and is git-ignored.

## Project Layout

```
zkllms/
├── zkllms/
│   ├── __init__.py
│   ├── cli.py             # click entry point, wires subcommands
│   ├── model.py           # decoder-LLM layer slice → ONNX export (any rotary causal LM)
│   ├── crypto.py          # AES-GCM encrypt/decrypt + PBKDF2 key derivation
│   ├── circuit.py         # ezkl: gen_settings → calibrate → compile → setup
│   ├── prover.py          # ezkl: gen_witness → prove
│   ├── verifier.py        # ezkl: verify
│   ├── backend.py         # runs ezkl's async calls inside an event loop
│   └── instrumentation.py # TEE computational-cost profiling
├── tests/                 # unit tests + slow (marked) model & end-to-end tests
├── Makefile               # install / test tiers / coverage / pipeline targets
├── pyproject.toml
└── README.md
```

## Development

A `Makefile` wraps setup, the test tiers, coverage, and the CLI pipeline. Run `make help`
for the full list.

```bash
make install      # create the virtualenv and install the package (editable) with dev deps
make test         # fast unit tests (excludes slow)
make test-slow    # slow tests: model download + real ezkl proving
make test-all     # entire suite
make coverage     # full suite with a term-missing coverage report
make run          # full pipeline: export -> setup -> prove -> verify
```

Pipeline variables are overridable, e.g. `make run MODEL_NAME=meta-llama/Llama-3.2-1B
SEQ_LEN=2 INPUT="the answer is"`.

Tests are split into two tiers via the `slow` marker (registered in `pyproject.toml`):

- **Fast** (`pytest -m "not slow"`) — pure logic and mocked ezkl/model calls; no network,
  no heavy compute. These are the unit tests.
- **Slow** (`pytest -m slow`) — download a model from HuggingFace and/or run the real ezkl
  `compile → setup → prove → verify` pipeline (the end-to-end test uses a small ONNX model
  so it verifies a real proof in seconds).

Development follows TDD (red-green-refactor); the code intentionally carries no inline
comments.

## Hardware Firmware Target

The verifier (and optionally the prover) is designed with hardware implementation in mind:

- halo2 uses a PLONK arithmetization — the verifier is a fixed arithmetic circuit,
  well-suited for synthesis into firmware
- verification keys (`vk`) can be exported and embedded in firmware
- the prover can run on GPU via the ICICLE backend (MSM / NTT acceleration)
- non-linear ops (softmax, GELU, layernorm) are handled by lookup tables inside the circuit

Because ezkl requires plaintext weights and inputs in RAM during proving, confidentiality
of the executing computation is delegated to a TEE on the accelerator: **ezkl guarantees
correctness, the TEE guarantees confidentiality.** This prototype simulates the TEE
boundary in-process; the `instrumentation` module profiles the computational cost of the
operations that would run inside it.

## Prototype scope

A faithful, fully-provable LLM forward pass is not tractable on CPU with today's zkML
tooling. Two deliberate simplifications keep this prototype provable in minutes rather
than hours while still exercising real pretrained weights (the concrete figures below are
for the default Qwen2.5-0.5B and scale with whichever `--model-name` you choose):

- **Transformer block only.** A full Qwen slice with the 151,936-entry embedding table
  and `lm_head` projection compiles to a ~1.1 GB ONNX graph and billions of constraints.
  Instead, the circuit proves the attention + feed-forward block on hidden states
  (`inputs_embeds → hidden_states`). The embedding lookup and `lm_head` projection are
  cheap linear maps that run on the host, on public values.
- **Bidirectional attention.** HuggingFace's causal masking fills masked positions with
  `-inf` and emits an `IsNan` op — neither is representable in ezkl's fixed-point field.
  The prototype uses a zero additive mask (full/bidirectional attention) for a single
  fixed-length forward pass. The full attention + FFN compute with real weights is still
  proven; the pass is not autoregressive.

The transformer block compiles to a valid halo2 circuit (`logrows` 20–23), but
generating its proving key (`setup`) and proof (`prove`) for the real 896-dimensional
Qwen layer needs substantially more RAM than a typical dev machine (observed to exceed
15 GB, at any `logrows`). Full proving therefore targets a GPU (ezkl's ICICLE backend)
or a high-memory host. The shipped end-to-end test (`tests/test_e2e.py`) drives the exact
same `circuit → setup → prover → verifier` code path with a small ONNX model, so the ezkl
integration is verified to produce a cryptographically valid, verifiable proof in seconds
without that memory pressure.

## References

- [ezkl](https://github.com/zkonduit/ezkl) — ONNX → ZK circuit compiler
- [halo2](https://zcash.github.io/halo2/) — PLONK-based proof system
- [ICICLE](https://github.com/ingonyama-zk/icicle) — GPU-accelerated ZK proving
- [Artemis](https://arxiv.org/abs/2409.12055) — commit-and-prove for private model weights
- [Qwen2.5](https://huggingface.co/Qwen/Qwen2.5-0.5B) — the base language model

## Comparison to ZK-DeepSeek

The most directly comparable published work is Y. Wang, *Zero-Knowledge Proof Based
Verifiable Inference of Models* (arXiv:2511.19902, 2025), which builds **ZK-DeepSeek** — a
SNARK-verifiable version of the full 671B-parameter DeepSeek-V3. It shares this project's
goal (prove inference correctness without revealing weights) but sits at the opposite end
of the effort/fidelity trade-off: it proves the entire model end-to-end, where `zkllms`
proves a single transformer block on hidden states.

| Dimension | zkllms (this project) | ZK-DeepSeek (Wang, 2025) |
|-----------|-----------------------|--------------------------|
| Proof system | halo2 + KZG via ezkl — universal **trusted-setup** SRS | Kimchi (PLONKish) + Pickles recursion, Pasta curves — **no trusted setup** |
| Matmul encoding | ezkl's generic ONNX lowering | CRPC/ZMul: `O(anb) → O(an+nb)` via Schwartz–Zippel; weight proof reused |
| Composition | single monolithic circuit | recursive: segments → binary-tree merge → per-layer → whole model |
| Scope proven | one block; embed + lm_head + causal mask excluded | full pipeline: embedding, MLA, MoE, RoPE, softmax, RMSNorm, SiLU, top-k |
| Attention | bidirectional (ezkl cannot represent `-inf`/`IsNan`) | full causal MLA + KV cache |
| Model | any rotary decoder LLM (default Qwen2.5-0.5B), one layer | DeepSeek-V3 671B, full MoE |
| Proof size / verify | ~7–22 KB / ms | 32–36 KB / ~350 ms, **constant** regardless of model size |
| Proving cost | minutes (small model); OOM for a full layer at 15 GB | hours per component (e.g. one `wkv_a1` matmul ≈ 57 h) |
| Model storage | ~60 MB ONNX block | 2.5 TB (Int64, from 680 GB BF16) |
| Hardware | CPU, 15 GB | i9 32-core + RTX 5090 + 64 GB RAM + 6 TB SSD + 800 GB swap |

**Advantages of ZK-DeepSeek:** no trusted setup; constant proof size and verification time
independent of model size (via recursive composition); full-fidelity coverage of causal
attention, MoE and all nonlinearities; a CRPC matmul scheme that sharply reduces constraint
counts and reuses weight-matrix proofs; demonstrated on a real state-of-the-art 671B model
and open-sourced.

**Disadvantages of ZK-DeepSeek:** proving is extremely expensive — hours per *component*
(a single matmul ≈ 57 h, softmax ≈ 11 h), so a full-model proof is not yet practical (the
authors name GPU acceleration as future work); Int64 quantization inflates the model to
2.5 TB and risks accuracy loss with no reported baseline comparison; every operator is a
bespoke, DeepSeek-specific circuit requiring large engineering effort; and it needs
high-end hardware even for the component benchmarks.

**Where this project differs:** `zkllms` favours off-the-shelf tooling (ezkl ONNX→circuit)
and a deliberately reduced scope to stay fast and simple, and it separates *correctness*
(the ZK layer) from *confidentiality of execution* (a simulated TEE), which the paper's
pure-ZK design does not require. The two ideas from ZK-DeepSeek most worth borrowing here
are its **recursive segment-merge** composition (which would dissolve the full-layer OOM by
never materialising the whole circuit) and its **integer-approximated softmax** (which
would let us restore causal attention without ezkl choking on `-inf`/`IsNan`).

- Y. Wang, *Zero-Knowledge Proof Based Verifiable Inference of Models*, arXiv:2511.19902,
  2025. https://arxiv.org/abs/2511.19902 — code: https://github.com/arcstar-lab/ZK-DeepSeek
