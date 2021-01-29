#ifndef __PLAQUETTE__
#define __PLAQUETTE__

#include "complex.h"
#include "array.h"


namespace U1{

void plaquette(double *lat, double *plaq);

complexd* Plaquette(Array<double> *dev_lat, complexd *plaq, bool print=false);



void PlaquetteFields(Array<double> *lat, Array<complexd> **plaqfield, Array<complexd> **plaq, bool spacetime, bool evenoddOrder);




void Fmunu(Array<double> *lat, Array<complexd> **fmunu_vol, Array<complexd> **fmunu);
}

#endif
