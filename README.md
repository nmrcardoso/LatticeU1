# Lattice U1

Code includes:
- Isotropic and Anisotropic lattice
- Multilevel for PP*
  - added generic with the number of time slices at level 0 and 1 choosen by the user
- Multilevel for PP*O (charges placed along z direction)
  - 1D, but 2D and 3D can be obtained by running the code several times passed the perpendiular points
  - to calculate (Ex², Ey², Ez², Bx², By², Bz²)
  - to calculate (Ex, Ey, Ez, Bx, By, Bz)  <-- Needs to be reviewed
  - added generic with the number of time slices at level 0 and 1 choosen by the user
- APE smearing (to check)
- Multihit smearing
- Wilson Loop
- Plaquette field
- Electric and Magnetic fields based on the Polyakov loop
  - 2D results in the charges plane
  - 2D results in the perpendicular plane in the middle of the charges
  - charges placed along z direction
  - to calculate (Ex², Ey², Ez², Bx², By², Bz²)
  - to calculate (Ex, Ey, Ez, Bx, By, Bz)  <-- Needs to be reviewed
- Electric and Magnetic fields based on the Wilson loop
  - 2D results in the charges plane
  - 2D results in the perpendicular plane in the middle of the charges
  - charges placed along z direction
  - to calculate (Ex², Ey², Ez², Bx², By², Bz²)
  - to calculate (Ex, Ey, Ez, Bx, By, Bz)  <-- Needs to be reviewed
- Auto kernel tunning

Before compiling, please change line 16 of Makefile for the correct GPU architecture:
GPU_ARCH = sm_XY

The code was primarly developed for 4D(3+1), however most of the code also supports 2D(1+1) and 3D(2+1).
