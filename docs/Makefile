# Minimal makefile for Sphinx documentation
#

# You can set these variables from the command line, and also
# from the environment for the first two.
SPHINXOPTS    ?=
SPHINXBUILD   ?= sphinx-build
SOURCEDIR     = source
USERGUIDEDIR  = userguide
GENAPIDIR     = generated_api
BUILDDIR      = build

# Put it first so that "make" without argument is like "make help".
help:
	@$(SPHINXBUILD) -M help "$(SOURCEDIR)" "$(BUILDDIR)" $(SPHINXOPTS) $(O)

.PHONY: help clean copy-examples copy-quickstart generate-userguide generate-api html Makefile

clean:
	@echo "[Makefile] cleaning..."
	@rm -rf "./$(BUILDDIR)"
	@rm -rf "./$(SOURCEDIR)/examples"
	@rm -rf "./$(SOURCEDIR)/generated_api"
	@rm -rf "./$(SOURCEDIR)/quickstart"
	@rm -rf "./$(SOURCEDIR)/userguide"
	@rm -rf "./$(SOURCEDIR)/release_notes"
	@rm -rf "./$(SOURCEDIR)/README.rst"

copy-examples:
	@echo "[Makefile] copying examples..."
	@cp -r ../examples "$(SOURCEDIR)/examples"
	@rm "$(SOURCEDIR)/examples/00-quickstart.ipynb"

copy-quickstart:
	@echo "[Makefile] copying quickstart notebook..."
	@mkdir -p "$(SOURCEDIR)/quickstart"
	@cp -r ../examples/00-quickstart.ipynb "$(SOURCEDIR)/quickstart/"
	@cp -r ../examples/static "$(SOURCEDIR)/quickstart/"

generate-release_notes:
	@echo "[Makefile] generating RELEASE_NOTES rst file..."
	@mkdir -p "$(SOURCEDIR)/release_notes"
	@m2r2 ../CHANGELOG.md
	@mv ../CHANGELOG.rst "$(SOURCEDIR)/release_notes/RELEASE_NOTES.rst"

generate-userguide:
	@echo "[Makefile] generating userguide rst files..."
	@find $(USERGUIDEDIR)/*.md -exec m2r2 {} \;
	@mkdir -p "$(SOURCEDIR)/userguide"
	@mv $(USERGUIDEDIR)/*.rst "$(SOURCEDIR)/userguide/"
	@cp -r "$(USERGUIDEDIR)/images" "$(SOURCEDIR)/userguide/"

generate-api:
	@echo "[Makefile] generating API using sphinx-apidoc..."
	# note all paths after "$(SOURCEDIR)/generated_api" will be excluded from rst generation
	@sphinx-apidoc -e -f --templatedir templates -o "$(SOURCEDIR)/generated_api" \
	    ../darts \
	    ../darts/logging.py \
	    ../darts/tests/* \
        ../darts/ad/aggregators/aggregators.py \
        ../darts/ad/anomaly_model/anomaly_model.py \
        ../darts/ad/detectors/detectors.py \
        ../darts/ad/scorers/scorers.py \
        ../darts/datasets/dataset_loaders.py \
        ../darts/explainability/explainability.py \
        ../darts/explainability/utils.py \
        ../darts/models/components/* \
        ../darts/models/filtering/filtering_model.py \
        ../darts/models/forecasting/ensemble_model.py \
        ../darts/models/forecasting/forecasting_model.py \
        ../darts/models/forecasting/foundation_model.py \
        ../darts/models/forecasting/torch_forecasting_model.py \
        ../darts/models/forecasting/pl_forecasting_module.py \
        ../darts/utils/data/tabularization/* \
        ../darts/utils/data/torch_datasets/dataset.py \
        ../darts/utils/data/torch_datasets/utils.py \
        ../darts/utils/data/utils.py \
        ../darts/utils/historical_forecasts/* \
        ../darts/utils/likelihood_models/base.py \
        ../darts/utils/multioutput.py \
        ../darts/utils/onnx_utils.py \
        ../darts/utils/ts_utils.py

html:
	# Note: this target has to be called "html" because its name is given as an argument to Sphinx
	@echo "[Makefile] generating HTML pages using sphinx-build..."
	@$(SPHINXBUILD) -M $@ "$(SOURCEDIR)" "$(BUILDDIR)" $(SPHINXOPTS) $(O)

build-all-docs: clean copy-examples copy-quickstart generate-release_notes generate-userguide generate-api html
build-api: clean generate-api html

# Catch-all target: route all unknown targets to Sphinx using the new
# "make mode" option.  $(O) is meant as a shortcut for $(SPHINXOPTS).
# %: Makefile
# 	@$(SPHINXBUILD) -M $@ "$(SOURCEDIR)" "$(BUILDDIR)" $(SPHINXOPTS) $(O)
