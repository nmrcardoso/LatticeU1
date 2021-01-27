#ifndef __SMEARING__
#define __SMEARING__

#include "complex.h"
#include "array.h"


namespace U1{


//option 0: Apply multihit only in space links
//option 1: Apply multihit only in time links
//option 2: Apply multihit all links
//option other: convert phase(double) to gauge link (complex double) or just copy if complex double
template<class Real>
Array<complexd>* ApplyMultiHit(Array<Real>* lat, int option);


//option 0: Apply multihit only in space links
//option 1: Apply multihit only in time links
//option 2: Apply multihit all links
template<class Real>
Array<Real>* ApplyAPE(Array<Real>* lat, double w, int n, int option);

}

#endif
