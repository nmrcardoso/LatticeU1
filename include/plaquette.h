#ifndef __PLAQUETTE__
#define __PLAQUETTE__

#include "complex.h"


namespace U1{

void plaquette(double *lat, double *plaq);

complexd dev_plaquette(double *dev_lat);

complexd dev_plaquette(double *dev_lat, complexd *dev_plaq, double norm, int threads, int blocks);

}

#endif
