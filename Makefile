.PHONY: all wheel clean
all: wheel
wheel:
	python3 setup.py bdist_wheel sdist
clean:
	rm -fr build dist *.egg-info

test:
	python -m unittest discover

docker: clean wheel docker-image

docker-image:
	docker build -t zetastitcher -f docker/Dockerfile dist
