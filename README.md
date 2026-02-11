# point-clouds-thesis


## Setup
After cloning, run:(requires clang-format)
```bash
git config core.hooksPath scripts/git-hooks
```

## Development Setup

Download and compile from source and install the PCL library.

`git clone https://github.com/PointCloudLibrary/pcl.git`

You will need to install the following libraries(and likely more if they are not installed on your system):

- Eigen (on most linux package managers it is probably there already)
- Boost (w. the devel package, libboost_iostreams)
- VTK (needed for visualisation)
- Flann (needed for search and visualisation)
- OpenCV (in package repositories most likely)


Configure it with CMake: 
`cmake -B build -S .`

Compile it(will take some time): 
`cd build && make`


`sudo make install`

