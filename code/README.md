# MICCAI 2015 Cell Detection using Proximity Score Regression
## Code Setup
We tested this code on Ubuntu/Debian.

### Pre-Processing
Compute the proximity score maps from dot-annotated images using the instructions in the `preprocessing` directory.

### Random Forest
#### Dependencies
* libconfig++ (1.4.9, 1.5.0)
* libeigen3-dev (>=3.1)
* libblitz (0.10)
* OpenCV (2.4.9)
* Boost (>= 1.54)
* GCC (<=4.8.5)
* CMake (>=2.6)

#### Setup
Compile the executable (class/regr) in the `bin` directory:
```bash
$ mkdir code/bin/celldetection-{class|regr}/build/
$ cd code/bin/celldetection-{class|regr}/build/
$ cmake ../../../src/apps/celldetection-{class|regr}/
$ make
```

#### Troubleshooting
Installing GCC 4.8 toolset:
```bash
$ sudo apt-get install -y gcc-4.8 g++-4.8 gfortran-4.8 
```

If you encounter issues when linking to `config++`, e.g. on newer Ubuntu (>=16.04), [download the libconfig source](http://www.hyperrealm.com/libconfig/libconfig-1.5.tar.gz) and build it by specifying the compiler you are going to use for the main program:
```bash
$ ./configure CC=gcc-4.8 CXX=g++-4.8
$ make 
$ sudo make install
```

Further, you may be required to compile boost 1.55 with gcc-4.8:
```bash
$ cd /opt
$ wget -O boost_1_55_0.tar.gz http://sourceforge.net/projects/boost/files/boost/1.55.0/boost_1_55_0.tar.gz/download
$ tar xzvf boost_1_55_0.tar.gz
$ cd boost_1_55_0/
$ sudo apt-get update
$ sudo apt-get install -y python-dev autotools-dev libicu-dev build-essential libbz2-dev 

# edit the build config
$ nano tools/build/v2/user-config.jam
# uncomment the line # using gcc ..., and change the version to 4.8
$ ./bootstrap.sh --prefix=/usr/local
# build and install using all CPU cores
$ sudo ./b2 --with=all --toolset=gcc -j4 install

# Reset the ldconfig
$ sudo ldconfig
```

The `cmake` line in the setup then becomes:
```bash
$ cmake -DCMAKE_C_COMPILER=gcc-4.8 -DCMAKE_CXX_COMPILER=gcc-4.8 ../../../src/apps/celldetection-{class|regr}/
```
### Post-Processing
* Matlab (tested with R2013a), add the `matlab` command to your system path
* Piotr's Computer Vision Toolbox (check out into `code` from https://github.com/pdollar/toolbox/)

