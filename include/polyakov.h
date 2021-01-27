#ifndef __POLYAKOV__
#define __POLYAKOV__

#include "complex.h"
#include "array.h"


namespace U1{

void polyakov(double *lat, double *poly);

//with tune
complexd Polyakov(Array<double> *dev_lat, bool print=false);

//With tune
template<class Real>
Array<complexd>* Poly2(Array<Real> *dev_lat, bool multihit);

}

#endif
