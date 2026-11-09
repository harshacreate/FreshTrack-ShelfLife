.PHONY: setup train test
setup:
	python3 -m pip install -e '.[test]'
train:
	python3 -m freshtrack.train
test:
	python3 -m pytest -q
