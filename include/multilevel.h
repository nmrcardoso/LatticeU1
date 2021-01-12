#ifndef __MULTILEVEL__
#define __MULTILEVEL__

#include "random.h"
#include "array.h"

namespace U1{

Array<complexd>* MultiLevel(Array<double> *dev_lat, CudaRNG *rng_state, int n4, int k4, int n2, int k2, int metrop, int ovrn);

}
#endif
