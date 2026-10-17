PYTHON ?= python3

.PHONY: all help profile benchmarks figures clean

all: profile benchmarks figures

help:
	@echo "Available commands:"
	@echo "  make profile     - Compile & run C99 embedded firmware profiling harness"
	@echo "  make benchmarks  - Run clinical benchmark evaluation suite"
	@echo "  make figures     - Regenerate vector & raster figures"
	@echo "  make clean       - Clean temporary build artifacts"

profile:
	$(MAKE) -C src/profiling run

benchmarks:
	$(PYTHON) src/benchmarks/run_benchmarks.py
	$(PYTHON) src/benchmarks/run_dynamic_benchmarks.py
	$(PYTHON) src/benchmarks/evaluate_robust_baselines.py
	$(PYTHON) src/benchmarks/generate_paper_macros.py

figures:
	$(PYTHON) src/benchmarks/generate_figures.py

clean:
	$(MAKE) -C src/profiling clean
	find . -type d -name "__pycache__" -exec rm -rf {} + 2>/dev/null || true
