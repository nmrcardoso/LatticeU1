#ifndef __PLAQUETTE__
#define __PLAQUETTE__

#include "complex.h"
#include "array.h"


namespace U1{

void plaquette(double *lat, double *plaq);

complexd Plaquette(Array<double> *dev_lat, bool print=true);

}

#endif
