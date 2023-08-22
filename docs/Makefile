# Minimal makefile for Sphinx documentation
#

# You can set these variables from the command line, and also
# from the environment for the first two.
SPHINXOPTS    ?=
SPHINXBUILD   ?= sphinx-build
SOURCEDIR     = source
BUILDDIR      = _build
GH_PAGES_SOURCES = . ../src/seismic Makefile

# Put it first so that "make" without argument is like "make help".
help:
	@$(SPHINXBUILD) -M help "$(SOURCEDIR)" "$(BUILDDIR)" $(SPHINXOPTS) $(O)

.PHONY: help Makefile

# Catch-all target: route all unknown targets to Sphinx using the new
# "make mode" option.  $(O) is meant as a shortcut for $(SPHINXOPTS).
%: Makefile
	@$(SPHINXBUILD) -M $@ "$(SOURCEDIR)" "$(BUILDDIR)" $(SPHINXOPTS) $(O)

gh-pages:
	git checkout gh-pages
	rm -rf $(BUILDDIR) _sources _static _modules chapters
	git checkout main $(GH_PAGES_SOURCES) .gitignore
	git reset HEAD
	jupyter nbconvert --to rst ../examples/*.ipynb
	cp -Rp ../examples/*.rst $(SOURCEDIR)/modules/tutorials/
	rm -rf ../examples/*.rst
	cp -Rp ../examples/*files $(SOURCEDIR)/modules/tutorials/
	rm -rf ../examples/*files
	make html
	cp -frv $(BUILDDIR)/html/* ./ && rm -R $(BUILDDIR)/html/*
	rm -rf $(GH_PAGES_SOURCES) $(BUILDDIR)
	git add -A
	git commit -m "Generated gh-pages for `git log main -1 --pretty=short --abbrev-commit`" && git push origin gh-pages ; git checkout main

dist:
	python3 -m pip install --user --upgrade setuptools wheel
	python3 setup.py sdist bdist_wheel
	python3 -m twine upload dist/*

