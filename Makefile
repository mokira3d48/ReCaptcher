venv:
	python3 -m venv env

install:
	pip install -r requirements.txt

dev:
	pip install -e .

test:
	pytest tests

run:
	python3 -m recaptcher

collect:
	python3 -m recaptcher configs/ds_collect.yaml

build-vocab:
	python3 -m recaptcher configs/ds_vocab.yaml

build-dataset:
	python3 -m recaptcher configs/ds_build.yaml
