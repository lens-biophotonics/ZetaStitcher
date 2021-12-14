.PHONY: all
all: wheel

.PHONY: wheel
wheel:
	python3 setup.py bdist_wheel sdist

.PHONY: clean
clean:
	rm -fr build dist *.egg-info

.PHONY: test
test:
	python -m unittest discover

.PHONY: docker
docker: clean wheel docker-image

.PHONY: docker-image
docker-image:
	docker build -t zetastitcher -f docker/Dockerfile dist
