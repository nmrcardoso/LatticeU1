#ifndef __UPDATE__
#define __UPDATE__


#include "random.h"
#include "array.h"

namespace U1{

void HotStart(Array<double> *dev_lat, CudaRNG *rng_state);

void UpdateLattice(Array<double> *dev_lat, CudaRNG *rng_state, int metrop, int ovrn);

}

#endif
