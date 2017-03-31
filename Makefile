.PHONY: all wheel clean
all: wheel
wheel:
	python3 setup.py bdist_wheel
clean:
	rm -fr build dist *.egg-info
