<p align="center">
<img src="https://raw.githubusercontent.com/lens-biophotonics/ZetaStitcher/master/doc/_static/zetastitcher.svg", height="150">
</p>

ZetaStitcher is a tool designed to stitch large volumetric images such as
those produced by Light-Sheet Fluorescence Microscopes.

Key features:

* able to handle datasets as big as 10<sup>12</sup> voxels
* multichannel images
* powerful and simple Python API to query arbitrary regions within the fused
volume

## How to install
On Ubuntu 20.04 LTS, run these commands:
```
sudo apt-get install python3-pip libgl1 libglib2.0-0
pip3 install zetastitcher
```

## Docker image
To build a docker image with ZetaStitcher:
```
make docker
```
You can call the stitching commands using an ephemeral container like this:
```
docker run -it -v`pwd`:/home --rm zetastitcher stitch-align -h
docker run -it -v`pwd`:/home --rm zetastitcher stitch-fuse -h
```

## Documentation
Please read the documentation and follow the tutorial at this page:<br/>
https://lens-biophotonics.github.io/ZetaStitcher/


## Acknowledgements
This open source software code was developed in whole in the Human
Brain Project, funded from the European Union’s Horizon 2020 Framework
Programme for Research and Innovation under Specific Grant Agreements
No. 720270 and No. 785907 (Human Brain Project SGA1 and SGA2).

<p align="center">
<img height="100" style="max-height: 100px" src="https://europa.eu/european-union/sites/europaeu/files/docs/body/flag_yellow_low.jpg">
Co-funded by the European Union
</p>
