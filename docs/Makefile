# Minimal makefile for Sphinx documentation
#

# You can set these variables from the command line, and also
# from the environment for the first two.
SPHINXOPTS    ?= 
SPHINXBUILD   ?= sphinx-build
SOURCEDIR     = source
USERGUIDEDIR  = userguide
BUILDDIR      = build

# Put it first so that "make" without argument is like "make help".
help:
	@$(SPHINXBUILD) -M help "$(SOURCEDIR)" "$(BUILDDIR)" $(SPHINXOPTS) $(O)

.PHONY: help Makefile

clean:
	@rm -rf "./$(BUILDDIR)"
	@rm -rf "./$(SOURCEDIR)/examples"
	@rm -rf "./$(SOURCEDIR)/generated_api"
	@rm -rf "./$(SOURCEDIR)/quickstart"
	@rm -rf "./$(SOURCEDIR)/userguide"
	@rm -rf "./$(SOURCEDIR)/README.rst"

copy-examples:
	@cp -r ../notebooks "$(SOURCEDIR)/examples"
	@rm "$(SOURCEDIR)/examples/00-quickstart.ipynb"

copy-quickstart:
	@mkdir -p "$(SOURCEDIR)/quickstart"
	@cp -r ../notebooks/00-quickstart.ipynb "$(SOURCEDIR)/quickstart/"

generate-readme:
	@m2r2 ../README.md
	@mv ../README.rst "$(SOURCEDIR)"

generate-userguide:
	find $(USERGUIDEDIR)/*.md -exec m2r2 {} \;
	@mkdir -p "$(SOURCEDIR)/userguide"
	@mv $(USERGUIDEDIR)/*.rst "$(SOURCEDIR)/userguide/"
	@cp -r "$(USERGUIDEDIR)/images" "$(SOURCEDIR)/userguide/"

generate-api:
	@sphinx-apidoc -e -f --templatedir templates -o "$(SOURCEDIR)/generated_api" ../darts ../darts/logging.py ../darts/tests/*

html:
	# Note: this target has to be called "html" because its name is given as an argument to Sphinx
	@$(SPHINXBUILD) -M $@ "$(SOURCEDIR)" "$(BUILDDIR)" $(SPHINXOPTS) $(O)

build-all-docs:
	clean
	copy-examples
	copy-quickstart
	generate-readme
	generate-userguide
	generate-api
	html

# Catch-all target: route all unknown targets to Sphinx using the new
# "make mode" option.  $(O) is meant as a shortcut for $(SPHINXOPTS).
# %: Makefile
# 	@$(SPHINXBUILD) -M $@ "$(SOURCEDIR)" "$(BUILDDIR)" $(SPHINXOPTS) $(O)
