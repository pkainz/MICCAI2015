# MICCAI 2015 Cell Detection using Proximity Score Regression
## Code Setup
We tested this code on Ubuntu/Debian.

### Pre-Processing
Compute the proximity score maps from dot-annotated images using the instructions in the `preprocessing` directory.

### Random Forest
#### Dependencies
* libconfig++
* libblitz (0.10)
* OpenCV (2.4.9)
* Boost (>= 1.54)
* GCC (<=4.8.5)
* CMake (>=2.6)

#### Setup
Compile the executable (class/regr) in the `bin` directory:
```bash
mkdir code/bin/celldetection-{class|regr}/build/
cd code/bin/celldetection-{class|regr}/build/
cmake ../../../src/apps/celldetection-{class|regr}/
make
```

### Post-Processing
* Matlab (tested with R2013a), add the `matlab` command to your system path
* Piotr's Computer Vision Toolbox (check out into `code` from https://github.com/pdollar/toolbox/)

