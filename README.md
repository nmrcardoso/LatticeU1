# Lattice U1

Code includes:
- Isotropic and Anisotropic lattice
- Multihit smearing
- Multilevel
- Auto kernel tunning

Before compiling, please change line 16 of Makefile
for the correct GPU architecture:
GPU_ARCH = sm_XY

The code was primarly developed for 4D(3+1), however most of this code supports also 2D(1+1), 3D(2+1).
