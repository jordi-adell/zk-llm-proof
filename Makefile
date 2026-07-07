VENV       ?= .venv
PY         := $(VENV)/bin/python
PIP        := $(VENV)/bin/pip
ZKLLMS     := $(VENV)/bin/zkllms

MODEL      ?= model.onnx
MODEL_NAME ?= Qwen/Qwen2.5-0.5B
KEYS       ?= keys
INPUT      ?= hello world
PROOF      ?= proof.json
SEQ_LEN    ?= 4
NUM_LAYERS ?= 1

export HF_HUB_DISABLE_PROGRESS_BARS = 1

.DEFAULT_GOAL := help

.PHONY: help install test test-fast test-slow test-all coverage \
        export setup prove verify run clean distclean

help:
	@echo "Setup"
	@echo "  make install      Create the virtualenv and install the package (editable) with dev deps"
	@echo ""
	@echo "Tests"
	@echo "  make test         Run the fast unit tests (default)"
	@echo "  make test-fast    Run the fast unit tests (excludes slow)"
	@echo "  make test-slow    Run the slow tests (Qwen download + real ezkl proving)"
	@echo "  make test-all     Run the entire suite"
	@echo "  make coverage     Run the full suite with a coverage report"
	@echo ""
	@echo "Pipeline (CLI steps)"
	@echo "  make export       Export the Qwen transformer block to ONNX"
	@echo "  make setup        Compile the circuit and generate proving/verification keys"
	@echo "  make prove        Run inference and generate a ZK proof"
	@echo "  make verify       Verify the proof"
	@echo "  make run          Run the whole pipeline: export -> setup -> prove -> verify"
	@echo ""
	@echo "  make clean        Remove generated artifacts (onnx, keys, proof)"
	@echo "  make distclean    Also remove the virtualenv"
	@echo ""
	@echo "Variables: MODEL_NAME=$(MODEL_NAME) MODEL=$(MODEL) KEYS=$(KEYS) SEQ_LEN=$(SEQ_LEN) NUM_LAYERS=$(NUM_LAYERS) INPUT='$(INPUT)'"

$(PY):
	python3 -m venv $(VENV) || python3 -m virtualenv $(VENV)

install: $(PY)
	$(PIP) install -U pip
	$(PIP) install -e ".[dev]"

test: test-fast

test-fast:
	$(PY) -m pytest -q -m "not slow"

test-slow:
	$(PY) -m pytest -q -m slow -p no:warnings

test-all:
	$(PY) -m pytest -q -p no:warnings

coverage:
	$(PY) -m pytest -q -p no:warnings --cov=zkllms --cov-report=term-missing

export:
	$(ZKLLMS) export --output $(MODEL) --model-name $(MODEL_NAME) --seq-len $(SEQ_LEN) --num-layers $(NUM_LAYERS)

setup:
	$(ZKLLMS) setup --model $(MODEL) --keys-dir $(KEYS)

prove:
	$(ZKLLMS) prove --model $(MODEL) --keys-dir $(KEYS) --input "$(INPUT)" --output $(PROOF) --model-name $(MODEL_NAME) --seq-len $(SEQ_LEN) --num-layers $(NUM_LAYERS)

verify:
	$(ZKLLMS) verify --keys-dir $(KEYS) --proof $(PROOF)

run: export setup prove verify

clean:
	rm -rf $(KEYS) $(MODEL) $(MODEL).calibration.json $(PROOF)

distclean: clean
	rm -rf $(VENV) *.egg-info .pytest_cache .coverage
