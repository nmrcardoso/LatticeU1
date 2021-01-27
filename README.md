# Lattice U1

Code includes:
- Isotropic and Anisotropic lattice
- Multilevel for PP*
- Multilevel for PP*O (charges placed along z direction)
- APE smearing (to check)
- Multihit smearing
- Wilson Loop
- Plaquette field
- Electric and Magnetic fields based on the Polyakov loop
  - 2D results in the charges plane
  - 2D results in the perpendicular plane in the middle of the charges
  - charges placed along y direction
- Electric and Magnetic fields based on the Wilson loop
  - 2D results in the charges plane
  - 2D results in the perpendicular plane in the middle of the charges
  - charges placed along y direction
- Auto kernel tunning

Before compiling, please change line 16 of Makefile for the correct GPU architecture:
GPU_ARCH = sm_XY

The code was primarly developed for 4D(3+1), however most of this code supports also 2D(1+1) and 3D(2+1).
