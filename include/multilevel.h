#ifndef __MULTILEVEL__
#define __MULTILEVEL__

#include "random.h"
#include "array.h"

namespace U1{

Array<complexd>* MultiLevel(Array<double> *dev_lat, CudaRNG *rng_state, int n4, int k4, int n2, int k2, int metrop, int ovrn);



void MultiLevelField(Array<double> *lat, CudaRNG *rng_state, Array<complexd> **pp, Array<complexd> **ppfield, int n4, int k4, int n2, int k2, int metrop, int ovrn);


Array<complexd>* MultiLevelTTO(Array<double> *lat, CudaRNG *rng_state, int n4, int k4, int n2, int k2, int metrop, int ovrn, int radius, bool SquaredField, bool alongCharges, bool symmetrize, int perpPoint = 0);


}
#endif
