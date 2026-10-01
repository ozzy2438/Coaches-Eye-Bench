# Coaches' Eye Bench. Two commands from a clean clone:  make data && make smoke
PY       := uv run
CEB      := $(PY) ceb
DEV      := $(shell $(CEB) seasons dev 2>/dev/null || echo 2012:2025)
TEST     := $(shell $(CEB) seasons test 2>/dev/null || echo 2026)

install-ci:  # copy the CI workflow into place (pushing it needs a token with the `workflows` scope)
	mkdir -p .github/workflows && cp ci/github-actions-ci.yml .github/workflows/ci.yml

.PHONY: install-ci help setup data data-test build build-test smoke smoke-real train eval-val eval-test \
        report card demo-card test lint check-protocol clean

help:
	@echo "setup | data | smoke | smoke-real | train | eval-val | eval-test | report | card | test | lint"

setup:
	uv sync

# ---- data (needs R + fitzRoy 1.8.0 and CEB_CONTACT_EMAIL; see DATA.md) -------------------------------
data:
	Rscript r/fetch.R --seasons $(DEV) --out data/raw
	$(CEB) manifest
	$(CEB) build

# the locked test season: refuses unless the protocol-frozen tag exists and protocol.md is unchanged
data-test: check-protocol
	Rscript r/fetch.R --seasons $(TEST) --out data/raw
	$(CEB) build --test

check-protocol:
	$(CEB) guard

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
