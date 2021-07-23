# Minimal makefile for Sphinx documentation
#

# You can set these variables from the command line, and also
# from the environment for the first two.
SPHINXOPTS    ?= 
SPHINXBUILD   ?= sphinx-build
SOURCEDIR     = source
BUILDDIR      = build

# Put it first so that "make" without argument is like "make help".
help:
	@$(SPHINXBUILD) -M help "$(SOURCEDIR)" "$(BUILDDIR)" $(SPHINXOPTS) $(O)

.PHONY: help Makefile

generate:
	@sphinx-apidoc -e -f --templatedir templates -o "$(SOURCEDIR)/generated_api" ../darts ../darts/logging.py ../darts/tests/*

readme:
	@m2r2 ../README.md; mv ../README.rst "$(SOURCEDIR)"

copy-examples:
	@cp -r ../examples "$(SOURCEDIR)/examples"

clean:
	@rm -rf ./build
	@rm -rf ./source/examples
	@rm -rf ./source/generated_api
	@rm -rf ./source/README.rst

# Catch-all target: route all unknown targets to Sphinx using the new
# "make mode" option.  $(O) is meant as a shortcut for $(SPHINXOPTS).
%: Makefile
	@$(SPHINXBUILD) -M $@ "$(SOURCEDIR)" "$(BUILDDIR)" $(SPHINXOPTS) $(O)
