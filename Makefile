.phony: test
test:
	pytest tests

.phony: dev_install
dev_install:
	bash ./install_dev_requirements.sh