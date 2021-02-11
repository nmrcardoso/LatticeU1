#ifndef __POLYAKOV__
#define __POLYAKOV__

#include "complex.h"
#include "array.h"


namespace U1{

void polyakov(double *lat, double *poly);

complexd Polyakov(Array<double> *dev_lat, bool print=false);


template<class Real>
Array<complexd>* Poly2(Array<Real> *dev_lat, uint Rmax, bool multihit);


template<class Real>
void Poly2(Array<Real> *lat, Array<complexd> **pp, Array<complexd> **ppspace, uint Rmax, bool multihit);

}

#endif
