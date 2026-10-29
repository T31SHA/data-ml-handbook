PY ?= python3
.PHONY: help install fonts test outputs figures notebooks docs serve pdf all clean

help:            ## list targets
	@grep -E '^[a-z]+:.*##' $(MAKEFILE_LIST) | awk -F':.*## ' '{printf "  %-11s %s\n", $$1, $$2}'
install:         ## install Python + Node deps
	$(PY) -m pip install -r requirements.txt -r requirements-build.txt
	npm ci --prefix build/mathjax
fonts:           ## download Fira Sans / IBM Plex Mono
	./build/fetch_fonts.sh
test:            ## execute every snippet + check includes
	$(PY) -m pytest -q
outputs:         ## re-run snippets and refresh code/out/*.txt
	$(PY) build/run_snippets.py
figures:         ## regenerate figures from figures/src
	cd figures/src && for f in charts2.py charts3.py charts4.py; do $(PY) $$f || exit 1; done
	$(PY) figures/src/flows.py
	cd code && HANDBOOK_FIG_DIR=../figures $(PY) proj02_eda.py >/dev/null && HANDBOOK_FIG_DIR=../figures $(PY) s03_viz.py >/dev/null
notebooks:       ## regenerate Colab notebooks
	$(PY) build/make_notebooks.py
docs:            ## build the MkDocs site into site/
	$(PY) build/make_docs.py && mkdocs build --strict
serve:           ## live-preview the docs
	$(PY) build/make_docs.py && mkdocs serve
build/mathjax/node_modules:
	npm ci --prefix build/mathjax
pdf: build/mathjax/node_modules  ## build dist/Data-ML-Handbook.pdf
	$(PY) build/build_pdf.py dist/Data-ML-Handbook.pdf
all: test docs pdf
clean:
	rm -rf dist site build/.cache docs/handbook docs/figures
