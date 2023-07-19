SRCS=orax

.PHONY: compile-requirements
compile-requirements:
	pip-compile --allow-unsafe --resolver=backtracking requirements.in
	pip-compile --allow-unsafe --resolver=backtracking requirements-dev.in

.PHONY: fmt
fmt:
	black $(SRCS)
	ruff --fix $(SRCS)

.PHONY: lint
lint:
	ruff $(SRCS)
	black --check --diff $(SRCS)

.PHONY: test
test:
	CUDA_VISIBLE_DEVICES='' pytest orax

.PHONY: test
clean:
	ruff clean
	find orax -regex '^.*\(__pycache__\|\.py[co]\)$$' -delete
