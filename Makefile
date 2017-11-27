.PHONY: all wheel clean
all: wheel
wheel:
	python3 setup.py bdist_wheel sdist
clean:
	rm -fr build dist *.egg-info
