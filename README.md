ZetaStitcher is a tool designed to sitch large volumetric images such as those produced by Light-Sheet Fluorescence Microscopes.


Key features:

* able to handle datasets as big as 10<sup>12</sup> voxels
* multichannel images
* powerful and simple Python API to query arbitrary regions within the fused volume

## Short guide
File names must be in the form X_Y_Z.tiff. There are two commands: `stitch-align` and `stitch-fuse` to be run in this order. Please refer to their help message (`-h` option) for more informations.

### stitch-align
```
usage: stitch-align [-h] [-o OUTPUT_FILE] [-c {r,g,b,s}] [-n N_OF_THREADS]
                    [--px-size-xy PX_SIZE_XY] [--px-size-z PX_SIZE_Z] --dz
                    MAX_DZ --dy MAX_DY --dx MAX_DX --overlap-h OH --overlap-v
                    OV [--z-samples ZSAMP] [--z-stride Z_STRIDE] [--iX] [--iY]
                    input_folder

Stitch tiles in a folder.

The following naming conventions are used:
* Z is the direction along the stack height,
* (X, Y) is the frame plane,
* y is the direction along which frames are supposed to overlap,
* x is the direction orthogonal to y in the frame plane (X, Y).

Unless otherwise stated, all values are expected in px.
    

positional arguments:
  input_folder          input folder

optional arguments:
  -h, --help            show this help message and exit
  -o OUTPUT_FILE        output file (default: stitch.yml)
  -c {r,g,b,s}          color channel (default: s)
  -n N_OF_THREADS       number of parallel threads to use (default: 8)

pixel size:
  If specified, the corresponding options can be expressed in your custom units.

  --px-size-xy PX_SIZE_XY
                        pixel size in the (X, Y) plane (default: 1)
  --px-size-z PX_SIZE_Z
                        pixel size in the Z direction (default: 1)

maximum shifts:
  --dz MAX_DZ           maximum allowed shift along Z (default: None)
  --dy MAX_DY           maximum allowed shift along y (the stitching axis)
                        relative to the nominal overlap (default: None)
  --dx MAX_DX           maximum allowed shift along x (lateral shift)
                        (default: None)

overlaps:
  --overlap-h OH        nominal overlap along the horizontal axis (default:
                        None)
  --overlap-v OV        nominal overlap along the vertical axis (default:
                        None)

multiple sampling along Z:
  Measure the optimal shift at different heights around the center of the stack

  --z-samples ZSAMP     number of samples to take along Z (default: 1)
  --z-stride Z_STRIDE   stride used for multiple Z sampling (default: None)

tile ordering:
  --iX                  invert tile ordering along X (default: False)
  --iY                  invert tile ordering along Y (default: False)

``` 

### stitch-fuse
```
usage: stitch-fuse [-h] [-o OUTPUT_FILENAME] [-d] [-c CHANNEL] [--zmin ZMIN]
                   [--zmax ZMAX] [-m | -a | -s] [-f] [--no-global] [--iX]
                   [--iY] [--px-size-xy PX_SIZE_XY] [--px-size-z PX_SIZE_Z]
                   yml_file

Fuse stitched tiles in a folder.

positional arguments:
  yml_file              .yml file produced by stitch align. It will also be
                        used for saving absolute coordinates. If a directory
                        is specifiedinstead of a file, uses a file named
                        "stitch.yml" if present. If the file does not exist,
                        it will be created (only where applicable: see option
                        -s).

optional arguments:
  -h, --help            show this help message and exit

output:
  -o OUTPUT_FILENAME    output file name. If not specified, no tiff output is
                        produced, only absoulute coordinates are computed.
                        (default: None)
  -d                    overlay debug info (default: False)
  -c CHANNEL            channel (default: -1)
  --zmin ZMIN
  --zmax ZMAX           noninclusive (default: None)

absolute positions:
  by default, absolute positions are computed by taking the maximum score in cross correlations

  -m                    take the maximum score in cross correlations (default)
                        (default: None)
  -a                    take the average result weighted by the score
                        (default: None)
  -s                    use nominal stage positions (default: None)
  -f                    force recomputation of absolute positions (default:
                        False)
  --no-global           do not perform global optimization (where applicable)
                        (default: False)

tile ordering (option -s only):
  --iX                  invert tile ordering along X (default: False)
  --iY                  invert tile ordering along Y (default: False)

pixel size:
  If specified, the corresponding options can be expressed in your custom units.

  --px-size-xy PX_SIZE_XY
                        pixel size in the (X, Y) plane (default: None)
  --px-size-z PX_SIZE_Z
                        pixel size in the Z direction (default: None)

```
