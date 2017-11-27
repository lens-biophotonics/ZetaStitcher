.. _tutorial:
Tutorial
=================
In this tutorial we will see how to use ZetaStitcher for stitching large 3D
datasets.

The dataset is expected to be composed of a collection of 3D stacks, with
adjacent stacks featuring an overlapping region in the XY plane. First of all,
make sure your files are named according to the following convention:
``X_Y.tiff`` where ``X`` and ``Y`` are the stage coordinates.

The following naming conventions are used:

* Z is the direction along the stack height,
* (X, Y) is the frame plane,
* y is the direction along which frames are supposed to overlap,
* x is the direction orthogonal to y in the frame plane (X, Y).

We will perform the stitching process in two steps:

1. we first run the command ``stitch-align`` in order to find the best
   alignment of all pairs of adjacent stacks. The results are saved to a small
   text file, usually named ``stitch.yml``
2. we then run the command ``stitch-fuse`` in order to produce the resulting
   fused 3D image.

Both commands can be run with the ``-h`` option to print a detailed help
message.

Nominal stitching
-----------------
Before we get started, it might be a good idea to produce a fused image where
the stacks are positioned at their nominal *stage* coordinates as specified in
the file names, i.e. we skip step 1 altogether for the moment. Open a terminal
and ``cd`` to the directory containing your collection of files. Then run the
following command::

    stitch-fuse -s --px-size-xy 0.5 --px-size-z 2 -o nominal200.tiff --zmin 200 --zmax 202 .

Let's break down the options used in the command above. The ``-s`` option tells
``stitch-fuse`` to use the nominal stage coordinates to produce the final
image. When working in this mode, it is mandatory to specify the pixel size
(this is needed to determine the size of the overlapping area between adjacent
stacks). We do so with the ``--px-size-xy`` and ``--px-size-z`` options. Pixel
sizes must be specified in the same units as the ``X``, ``Y`` coordinates in
the file name (say, microns). The input stacks are read in the order of
increasing ``X`` and ``Y``. If this is not the case (i.e. one or more axes of
your translational stage are inverted), use options ``--iX`` and ``--iY`` to
specify axis inversion. We then specify an output file name using the ``-o``
option. Finally, for this testing run, we don't want to produce the whole
output stack. We therefore limit the output between 200 and 202 um
(noninclusivee) using the ``--zmin`` and ``-zmax`` options. Since we have
specified a voxel size of 2 along Z, the output image will contain one frame
only (i.e. we are producing a 2D image). Note the trailing dot in the command
line: that's the path to the directory containing the input stacks (in this
case, that means the current directory). You can now inspect the result of this
*nominal stitching*.


Global optimal alignment and fusion
-----------------------------------
We now proceed with the complete alignment and fusion pipeline.

First, we compute pairwise optimal alignment of adjacent stacks::

    stitch-align --px-size-xy 0.5 --px-size-z 2 --overlap-h 40 --overlap-v 40 --dx 10 --dy 10 --dz 8 --z-samples 10 --z-stride 50 .

We are again specifying the voxel size for our own convenience. This is not
mandatory; if omitted, subsequent options are expected to be specified in pixel
units. As described above, use ``--iX`` and ``--iY`` if you need to invert one
axis compared to what is specified in the file names. We then specify the
nominal overlap along the horizontal and vertical directions (i.e. X and Y).
This specifies the size of the nominal overlapping area that adjacent tiles
are *expected* to share. Relative to this nominal overlap, the program will try
to find the best alignment within a region specified by these options:
``--dx``, ``--dy``, ``--dz``. Note the lowercase letters: ``--dy`` specifies
the maximum allowed shift *along the stitching axis*, whereas ``--dx``
specifies the maximum allowed lateral shift. For example, if two tiles are being
stitched horizontally (i.e. along X) and we have specified a ``--dy`` of 10um,
then the optimal position of the second tile will be searched in a region that
is 20um wide along X, centered on top of the nominal overlap. In the
transversal direction (i.e. the vertical direction, Y), the search range is
also +/- 10 microns as specified by ``--dx``. Finally we need to specify at how
many depths along the stack we want the best alignment to be computed. The
``--z-samples`` and ``--z-stride`` options that are used in the command above
mean that we want to take 10 samples along the stack depth (5 above and 5
below the stack center) at 50um of distance from one another. The trailing dot
in the command line is again the path to the current directory.

The command above prints some progress information and creates a file named
``stitch.yml`` in the current directory. We now want to inspect the result of
the alignment by taking a look at a fused frame::

    stitch-fuse -o test200.tiff --zmin 200 --zmax 202 .

This will save the frame located at a depth of 200um in the stack to a file
named ``test200.tiff``. Note that there is no need to specify the pixel size
again, since that information is stored in the yml file from the previous
command. The first time that the program is run, a global optimization
algorithm (simulated annealing) is run in order to optimize the position of all
stacks within the mosaic. If we are happy with the result, we can proceed and
fuse the whole volume with the following command::

    stitch-fuse -o fused.tiff .


Accessing the fused volume programmatically
-------------------------------------------
If the resulting fused volume is too big, such that it is impractical to
produce the output file, consider accessing portions of your volume
programmatically. To do so, use the `VirtualFusedVolume` class:

>>> from zetastitcher import VirtualFusedVolume
>>> vfv = VirtualFusedVolume('stitch.yml')
>>> vfv.shape
(2955, 15738, 18963)
>>> subvolume = vfv[1500, 15500:16000, 18000:19500]

Axis order is ZYX.

The ``subvolume`` variable is a :class:`numpy.ndarray` that you can use as
you like. If you want to save the subvolume to a tiff file:

>>> import skimage.external.tifffile as tiff
>>> tiff.imsave('subvolume.tiff', subvolume)
