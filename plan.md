# Plan: zkllms — Zero-Knowledge Proofs for LLM Inference on Hardware Accelerators

## Context

Build a Python CLI prototype demonstrating ZKP (PLONK via halo2) for transformer inference where model weights and input data are provided encrypted to a hardware accelerator. The hardware computes inference and produces a SNARK proof that the computation was correct — without revealing weights or inputs to external verifiers. ezkl is the chosen ZK compiler (ONNX → halo2 circuit). The CLI simulates the TEE/HW boundary in-process; the ZK math is real.

---

## Final Architecture

```
  Client (CLI)
  ┌─────────────────────────────────────────┐
  │ AES-GCM encrypt(weights, key) → c_w    │
  │ AES-GCM encrypt(input,   key) → c_x    │
  └─────────────────┬───────────────────────┘
                    │ (c_w, c_x)  [simulated secure channel]
  Hardware Accelerator [TEE boundary — simulated in-process]
  ┌─────────────────▼───────────────────────┐
  │ decrypt(c_w) → weights  (never leaves)  │
  │ decrypt(c_x) → input    (never leaves)  │
  │ ezkl gen_witness + prove → (output, π) │
  └─────────────────┬───────────────────────┘
                    │ (output, π)
  Verifier
  ┌─────────────────▼───────────────────────┐
  │ ezkl.verify(vk, π, output) → bool      │
  └─────────────────────────────────────────┘
```

ezkl visibility config:
- `param_visibility = "private"` — weights hidden from proof transcript
- `input_visibility = "hashed"` — Poseidon hash of input is public, plaintext is not
- `output_visibility = "public"` — output logits are public

---

## ezkl API Workflow (confirmed correct order)

```
gen_settings → calibrate_settings → compile_circuit → get_srs → setup → gen_witness → prove → verify
```

`get_srs` downloads a KZG trusted setup file (~10 MB). This requires internet access; integration tests that call it should be marked `slow`/skippable in CI.

---

## Project Layout

```
zkllms/
├── zkllms/
│   ├── __init__.py
│   ├── cli.py             # click entry point, wires subcommands
│   ├── model.py           # tiny 1-layer transformer → ONNX export
│   ├── crypto.py          # AES-GCM encrypt/decrypt + PBKDF2 key derivation
│   ├── circuit.py         # ezkl: gen_settings → calibrate → compile → setup
│   ├── prover.py          # ezkl: gen_witness → prove (calls instrumentation)
│   ├── verifier.py        # ezkl: verify
│   └── instrumentation.py # TEE cost profiling (time + memory per phase)
├── tests/
│   ├── test_model.py
│   ├── test_crypto.py
│   ├── test_circuit.py
│   ├── test_prover.py
│   ├── test_verifier.py
│   ├── test_instrumentation.py
│   └── test_e2e.py        # full pipeline integration test
├── README.md
└── pyproject.toml
```

---

## Implementation Steps

### Step 0 — README.md

Write first (establishes intent). Include: overview, architecture diagram, quickstart, project layout, hardware firmware notes, references.

### Step 1 — pyproject.toml

```toml
[project]
name = "zkllms"
version = "0.1.0"
dependencies = [
    "ezkl>=10.0.0",
    "torch>=2.0",
    "onnx>=1.14",
    "transformers>=4.40",   # Qwen2.5-0.5B model + tokenizer
    "click>=8.0",
    "cryptography>=41.0",
    "psutil>=5.9",          # cross-platform RSS measurement for TEE cost profiling
]

[project.optional-dependencies]
dev = ["pytest", "pytest-mock"]

[project.scripts]
zkllms = "zkllms.cli:cli"
```

Pin ezkl to `>=10.0.0` — the API changed significantly at v10 (async functions became sync, `ezkl.gen_settings()` signature changed). If the latest stable is older, use whatever is current and note the exact version in a comment.

### Step 2 — `zkllms/crypto.py`

Simple AES-GCM wrapper simulating the hardware encrypted channel. No external calls.

```python
# encrypt_weights(path: Path, key: bytes) → (ciphertext: bytes, nonce: bytes)
# decrypt_weights(ciphertext: bytes, nonce: bytes, key: bytes) → bytes
# derive_key(passphrase: str, salt: bytes) → bytes   # PBKDF2-HMAC-SHA256, 100k iterations
# generate_key() → bytes                             # os.urandom(32)
```

Unit tests mock nothing — pure crypto, no I/O.

### Step 3 — `zkllms/model.py`

Uses **Qwen2.5-0.5B** (Qwen/Qwen2.5-0.5B on HuggingFace) — a real 494M-parameter LLM with 24 layers, d_model=896, n_heads=14.

**Circuit size caveat**: A full 24-layer Qwen2.5-0.5B at seq_len=8 generates an extremely large ZK circuit (billions of constraints), making CPU proving impractical (hours+). The plan addresses this with a `--num-layers` flag that exports only the first N transformer layers (default: 1). This gives a real Qwen2.5-0.5B architecture slice with authentic weight distributions, while keeping circuit size manageable (~30–100M constraints per layer at seq_len=4).

```python
def load_model(num_layers: int = 1) -> tuple[nn.Module, PreTrainedTokenizer]:
    # AutoModelForCausalLM.from_pretrained("Qwen/Qwen2.5-0.5B")
    # Wraps model.model.layers[:num_layers] in a thin nn.Module that runs
    # embedding → first N decoder layers → lm_head (skipping remaining layers)
    # Returns (wrapped_model, tokenizer)

def export_to_onnx(
    model: nn.Module,
    tokenizer: PreTrainedTokenizer,
    path: Path,
    seq_len: int = 4,
) -> None:
    # torch.onnx.export with opset_version=17, dynamic_axes=None (fixed shape for ezkl)
    # Input: input_ids of shape (1, seq_len) — no KV cache, no attention_mask for simplicity
    # Output: logits of shape (1, seq_len, vocab_size)

def create_sample_input(tokenizer: PreTrainedTokenizer, text: str, seq_len: int = 4) -> dict:
    # Tokenizes text, pads/truncates to seq_len
    # Returns {"input_ids": [[...]]} — ezkl witness input JSON format
```

**Note**: Do NOT manually quantize. ezkl's `calibrate_settings` handles quantization internally. Export standard float32. The KV-cache-free single-pass export is intentional — ezkl requires static graphs.

### Step 4 — `zkllms/circuit.py`

Wraps the ezkl compilation pipeline. All ezkl calls may be async in older versions — use `asyncio.run()` if needed, or call synchronously in v10+.

```python
def compile_model(
    onnx_path: Path,
    input_json: Path,       # sample witness input for calibration
    settings_path: Path,
    circuit_path: Path,     # output compiled .ezkl circuit
) -> None:
    # ezkl.gen_settings(onnx_path, settings_path)
    # ezkl.calibrate_settings(settings_path, onnx_path, input_json, target="resources")
    # ezkl.compile_circuit(onnx_path, circuit_path, settings_path)

def setup_keys(
    circuit_path: Path,
    srs_path: Path,         # downloaded KZG SRS file
    pk_path: Path,
    vk_path: Path,
    settings_path: Path,
) -> None:
    # ezkl.get_srs(srs_path, settings_path)   ← requires internet
    # ezkl.setup(circuit_path, vk_path, pk_path, srs_path)
```

Unit tests mock `ezkl.*` calls with `pytest-mock`. Integration tests (marked `@pytest.mark.slow`) run the full pipeline.

### Step 5 — `zkllms/instrumentation.py`

Profiles the operations that run inside the simulated TEE boundary. Focuses on **computational work**, not time alone — useful for estimating feasibility on target firmware (GPU / FPGA / ASIC).

```python
@dataclass
class PhaseMetrics:
    name: str
    wall_s: float           # wall-clock seconds (time.perf_counter)
    cpu_s: float            # CPU user+sys time (time.process_time) — measures actual compute
    cpu_util_pct: float     # cpu_s / wall_s * 100 — shows parallelism efficiency
    peak_rss_mb: float      # peak resident memory delta (psutil.Process.memory_info)

@dataclass
class TEEProfile:
    phases: list[PhaseMetrics]
    proof_size_bytes: int   # size of the proof JSON on disk
    constraint_count: int   # from ezkl settings.json "num_rows" (circuit size)
    inference_flops: int    # theoretical FLOPs for transformer forward pass (analytical)

    def summary_table(self) -> str:
        # Returns a formatted table for CLI display — no mention of cost/money
        # Phase | Wall (s) | CPU (s) | CPU util% | Peak RAM (MB)
        # ...
        # ─────────────────────────────────────────────────────
        # Circuit constraints: 3,145,728
        # Proof size:          6.9 KB
        # Inference FLOPs:     ~12.6 M
        # (note: FLOPs inform ASIC/FPGA gate budget, not wall time)

@contextmanager
def tee_phase(name: str, phases: list[PhaseMetrics]):
    # Snapshots time.perf_counter() + time.process_time() + psutil RSS before/after
    # Appends PhaseMetrics to phases on exit

def estimate_transformer_flops(d_model: int, n_heads: int, seq_len: int, n_layers: int, intermediate: int) -> int:
    # Analytical FLOP estimate for Qwen-style decoder layer:
    #   attention: 4 * seq_len * d_model^2 + 2 * seq_len^2 * d_model
    #   FFN:       2 * seq_len * d_model * intermediate * 2  (SwiGLU = 2 matmuls)
    #   total per layer * n_layers
    # Qwen2.5-0.5B params: d_model=896, n_heads=14, intermediate=4864
    # Returns integer FLOPs — used to populate TEEProfile.inference_flops
```

`constraint_count` is read from the ezkl `settings.json` output field `"num_rows"` (populated after `calibrate_settings`). This maps directly to the ZK circuit size and verifier gate budget.

`cpu_util_pct` distinguishes single-threaded work (100%) from parallelized work (>100% on multi-core or <100% if I/O-bound) — both matter for firmware targeting.

### Step 6 — `zkllms/prover.py`

```python
@dataclass
class ProofResult:
    proof_path: Path
    output_token_ids: list[int]   # argmax over logits from proof["instances"]
    profile: TEEProfile | None

def generate_proof(
    circuit_path: Path,
    input_json: Path,
    pk_path: Path,
    srs_path: Path,
    witness_path: Path,
    proof_path: Path,
    settings_path: Path,
    profile: bool = False,
) -> ProofResult:
    phases = []
    with tee_phase("gen_witness", phases):
        ezkl.gen_witness(input_json, circuit_path, witness_path)
    with tee_phase("prove", phases):
        ezkl.prove(witness_path, pk_path, proof_path, srs_path, circuit_path)
    # Extract inference result from proof public instances
    proof_data = json.loads(proof_path.read_text())
    output_token_ids = _decode_instances(proof_data["instances"])  # argmax over quantized logits
    tee_profile = None
    if profile:
        constraint_count = _read_constraint_count(settings_path)
        proof_size = proof_path.stat().st_size
        tee_profile = TEEProfile(phases, proof_size, constraint_count, inference_flops=0)
    return ProofResult(proof_path, output_token_ids, tee_profile)
```

The `decrypt` phase is measured by wrapping `crypto.decrypt_weights()` + `crypto.decrypt_bytes()` in the CLI `prove` command with `tee_phase("decrypt", phases)` before calling `generate_proof()`. The phases list is passed in from the CLI and merged into the returned `TEEProfile`.

### Step 7 — `zkllms/verifier.py`

```python
def verify_proof(
    proof_path: Path,
    vk_path: Path,
    settings_path: Path,
    srs_path: Path,
) -> bool:
    # return ezkl.verify(proof_path, settings_path, vk_path, srs_path)
```

### Step 8 — `zkllms/cli.py`

Four subcommands via `@cli.command()`. Use `click.Path(exists=True)` for inputs, `click.Path()` for outputs.

```
zkllms export  --output model.onnx
               --seq-len 4        (default: 4 — shorter = fewer constraints)
               --num-layers 1     (default: 1 — exports first N Qwen2.5-0.5B decoder layers)
               Downloads Qwen/Qwen2.5-0.5B and exports N-layer slice to ONNX

zkllms setup   --model model.onnx
               --keys-dir keys/
               Compiles circuit and generates pk/vk/settings

zkllms prove   --model model.onnx
               --keys-dir keys/
               --input "Hello world"
               --output proof.json
               [--passphrase TEXT]        # used for AES-GCM encrypt of weights+input
               [--inference-output PATH]  # optional: save decoded inference result to file

zkllms verify  --keys-dir keys/
               --proof proof.json
               Prints "Proof verified ✓" or exits nonzero
```

The `prove` command outputs **both** the proof and the inference result:

```
Inference result
────────────────────────────────────────
Input:   "Hello world"
Output:  "Hello world, I am a language"
Tokens:  [9439, 1879, 11, 358, 1079, 264, 4128]
────────────────────────────────────────
Proof written to proof.json
```

The output is decoded from the proof's public `instances` field (ezkl embeds public outputs in the proof). Implementation:
1. After `ezkl.prove()`, load `proof.json` and read `proof["instances"]` — these are the quantized output logits
2. `argmax` over the vocab dimension to get predicted token IDs
3. Decode with the Qwen2.5 tokenizer
4. Print decoded text to stdout; optionally save raw logits to `--inference-output`

The `prove` command has a `--profile / --no-profile` flag (default: on). When enabled it also prints the `TEEProfile.summary_table()`:

```
TEE Firmware Computational Cost
──────────────────────────────────────────────────────────────
Phase         Wall (s)   CPU (s)   CPU util%   Peak RAM (MB)
decrypt          0.001     0.001       100%            0.01
gen_witness      4.823     9.421       195%           312.4
prove           11.240    44.812       399%           891.2
──────────────────────────────────────────────────────────────
Circuit constraints:  3,145,728   ← ZK gate budget (verifier)
Proof size:               6.9 KB
Inference FLOPs:        ~186.4 M   ← Qwen2.5-0.5B layer×1, seq_len=4
```

The `prove` command flow:
1. Loads the ONNX model, AES-GCM encrypts it (simulated HW channel), then decrypts to RAM
2. Tokenizes `--input` string using the Qwen2.5 tokenizer, pads/truncates to `seq_len`
3. Writes input JSON, calls `generate_proof()`, reads output from proof instances, decodes

---

## Dependency Between Modules

```
crypto.py            (no deps)
model.py             (torch, onnx, transformers — downloads Qwen2.5-0.5B on first call)
instrumentation.py   (no deps — stdlib only + psutil)
circuit.py           → ezkl
prover.py            → ezkl, instrumentation
verifier.py          → ezkl
cli.py               → model, circuit, prover, verifier, crypto, instrumentation
```

Write and test in that order. Each module is independently testable before cli.py is built.

---

## Testing Strategy

**Unit tests** (`test_model.py`, `test_crypto.py`): no mocking needed, pure logic.

**Unit tests with mocks** (`test_circuit.py`, `test_prover.py`, `test_verifier.py`): mock `ezkl.*` functions using `pytest-mock` `mocker.patch("zkllms.circuit.ezkl.gen_settings")` etc. Verify correct call order and argument passing.

**Instrumentation tests** (`test_instrumentation.py`): verify `tee_phase` records non-negative wall times, `TEEProfile.summary_table()` output contains expected column headers, and `proof_size_bytes` reads the actual file size.

**Integration test** (`test_e2e.py`): marked `@pytest.mark.slow`, runs the full pipeline end-to-end. Requires internet (SRS download). Can be skipped in CI with `pytest -m "not slow"`.

---

## Verification

```bash
# Fast unit tests (no internet, no ezkl compute)
pytest tests/ -v -m "not slow"

# Full end-to-end (requires internet + ~15s CPU proof time)
pytest tests/test_e2e.py -v -s

# CLI e2e
pip install -e ".[dev]"
zkllms export --output /tmp/model.onnx
zkllms setup --model /tmp/model.onnx --keys-dir /tmp/keys/
zkllms prove --model /tmp/model.onnx --keys-dir /tmp/keys/ --input "hello" --output /tmp/proof.json
zkllms verify --keys-dir /tmp/keys/ --proof /tmp/proof.json
# Expected: "Proof verified ✓"
```
