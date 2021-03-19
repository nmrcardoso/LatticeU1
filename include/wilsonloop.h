#ifndef __WILSONLOOP__
#define __WILSONLOOP__

#include "complex.h"
#include "array.h"


namespace U1{

template<class Real>
Array<complexd>* WilsonLoop(Array<Real>* lat, int R, int T, bool FastVersion=true);

template<class Real>
void WilsonLoop(Array<Real>* lat, Array<complexd>** wl, Array<complexd>** wlfield, int R, int T, bool FastVersion);

  
template<class Real>
Array<complexd>* WilsonLoopSS(Array<Real>* lat, int R, int T);


}

#endif
