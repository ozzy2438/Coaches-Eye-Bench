# Coaches' Eye Bench. Two commands from a clean clone:  make data && make smoke
PY       := uv run --frozen
CEB      := $(PY) ceb
DEV      = $(shell $(CEB) seasons dev)
TEST     = $(shell $(CEB) seasons test)
export R_LIBS_USER ?= $(CURDIR)/.r-library

restore-tag:  # recreate the protocol-frozen tag on a fresh clone (tag pushes were blocked in the build sandbox)
	bash scripts/restore_protocol_tag.sh

install-ci:  # copy the CI workflow into place (pushing it needs a token with the `workflows` scope)
	mkdir -p .github/workflows && cp ci/github-actions-ci.yml .github/workflows/ci.yml

.PHONY: restore-tag install-ci help setup setup-r data data-pilot data-test build build-test smoke smoke-real train eval-val eval-test \
        report card demo-card test lint check-protocol check-test clean

help:
	@echo "setup | data | smoke | smoke-real | train | eval-val | eval-test | report | card | test | lint"

setup:
	uv sync --frozen

setup-r:
	Rscript r/setup.R

# ---- data (needs R + fitzRoy 1.8.0 and CEB_CONTACT_EMAIL; see DATA.md) -------------------------------
data:
	Rscript r/fetch.R --seasons $(DEV) --out data/raw
	$(CEB) manifest
	$(CEB) build

# Independent real-data pilot; its model choices must never become study choices.
data-pilot:
	Rscript r/fetch.R --seasons 2024:2025 --out data/raw
	CEB_RAW_DIR=data/raw CEB_DATA_DIR=data/pilot CEB_RESULTS_DIR=results/pilot $(CEB) build --seasons 2024 2025
	CEB_RAW_DIR=data/raw CEB_DATA_DIR=data/pilot CEB_RESULTS_DIR=results/pilot $(CEB) smoke-real

build:
	$(CEB) build

build-test: check-test
	$(CEB) build --test

# the locked test season: refuses unless the protocol-frozen tag exists and protocol.md is unchanged
data-test: check-test
	Rscript r/fetch.R --seasons $(TEST) --out data/raw
	$(CEB) build --test

check-protocol:
	$(CEB) guard

check-test:
	$(CEB) guard --test

# ---- offline end-to-end on a synthetic fixture (< 2 min; output is never a result) -------------------
smoke:
	$(CEB) smoke --out results/smoke

# plumbing check on two real dev seasons already fetched by `make data`
smoke-real:
	$(CEB) smoke-real

# ---- modelling ---------------------------------------------------------------------------------------
train:            # model selection on validation only -> results/val/frozen_choices.json
	$(CEB) train

eval-val:
	$(CEB) eval-val

eval-test: check-protocol data-test   # single guarded run on the test season
	$(CEB) eval-test

report:
	$(CEB) report

card:             # local Match Review Card from your own eval-test output
	$(CEB) card --scores data/derived/test_scores.parquet --season $(TEST) --round $(ROUND) --out results/card_$(TEST)_R$(ROUND).html

demo-card: smoke
	$(CEB) card --scores results/smoke/derived/test_scores.parquet --season 2025 --round 3 --out docs/demo_card_synthetic.html --synthetic

test:
	$(PY) pytest -q

lint:
	$(PY) ruff check src tests

clean:
	rm -rf results/smoke results/smoke_real .pytest_cache .ruff_cache
