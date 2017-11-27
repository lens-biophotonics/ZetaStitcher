from setuptools import setup, find_packages


version = {}
with open("zetastitcher/version.py") as fp:
    exec(fp.read(), version)
__version__ = version['__version__']

setup(
    name='zetastitcher',
    version=__version__,
    description='Stitch large volumetric images (3D stacks)',
    long_description='ZetaStitcher is a tool designed to stitch large '
                     'volumetric images such as those produced by Light-Sheet '
                     'Fluorescence Microscopes.',
    author='Giacomo Mazzamuto',
    author_email='mazzamuto@lens.unifi.it',
    license='GPLv3+',
    # See https://pypi.python.org/pypi?%3Aaction=list_classifiers
    classifiers=[
        # How mature is this project? Common values are
        #   3 - Alpha
        #   4 - Beta
        #   5 - Production/Stable
        'Development Status :: 3 - Alpha',

        # Indicate who your project is intended for
        'Intended Audience :: Science/Research',
        'Topic :: Scientific/Engineering',

        # Pick your license as you wish (should match "license" above)
        'License :: OSI Approved :: GNU General Public License v3 or later (GPLv3+)',

        # Specify the Python versions you support here. In particular, ensure
        # that you indicate whether you support Python 2, Python 3 or both.
        'Programming Language :: Python :: 3.3',
        'Programming Language :: Python :: 3.4',
        'Programming Language :: Python :: 3.5',
    ],

    # What does your project relate to?
    keywords='stitch, microscopy',

    # You can just specify the packages manually here if your project is
    # simple. Or you can use find_packages().
    packages=find_packages(),

    # Alternatively, if you want to distribute just a my_module.py, uncomment
    # this:
    #   py_modules=["my_module"],

    # List run-time dependencies here.  These will be installed by pip when
    # your project is installed. For an analysis of "install_requires" vs pip's
    # requirements files see:
    # https://packaging.python.org/en/latest/requirements.html
    install_requires=[
        'coloredlogs',
        'networkx',
        'numpy',
        'pandas',
        'psutil',
        'pyfftw',
        'pygmo>=2',
        'pyyaml',
        'scipy',
        'scikit-image',
    ],

    # List additional groups of dependencies here (e.g. development
    # dependencies). You can install these using the following syntax,
    # for example:
    # $ pip install -e .[dev,test]
    extras_require={
        'dev': [
            'pip-tools',
        ],
        'dcimg': [
            'dcimg>=0.4.0'
        ],
        'doc': [
            'numpydoc',
            'sphinx',
            'sphinx_rtd_theme',
        ],
        'test': ['mock'],
    },

    # If there are data files included in your packages that need to be
    # installed, specify them here.  If using Python 2.6 or less, then these
    # have to be included in MANIFEST.in as well.
    package_data={
    },

    # To provide executable scripts, use entry points in preference to the
    # "scripts" keyword. Entry points provide cross-platform support and allow
    # pip to create the appropriate form of executable for the target platform.
    entry_points={
        'console_scripts': [
            'stitch-align = zetastitcher.runner:main',
            'stitch-fuse = zetastitcher.fuser.__main__:main',
        ],

    },
)
