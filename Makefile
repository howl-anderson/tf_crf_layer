.phony: test
test:
	pytest tests

.phony: dev_install
dev_install:
	bash ./install_dev_requirements.sh

release: dist ## package and upload a release
	twine upload dist/*

dist: clean ## builds source and wheel package
	python setup.py sdist
	python setup.py bdist_wheel
	ls -l dist

.PHONY: update_minor_version
update_minor_version:
	bumpversion minor

.PHONY: update_patch_version
update_patch_version:
	bumpversion patch

.PHONY: update_major_version
update_major_version:
	bumpversion major
