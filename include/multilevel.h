#ifndef __MULTILEVEL__
#define __MULTILEVEL__

#include "random.h"
#include "array.h"

namespace U1{

Array<complexd>* MultiLevel(Array<double> *dev_lat, CudaRNG *rng_state, int n4, int k4, int n2, int k2, int metrop, int ovrn, int Rmax, bool PrintResultsAtEveryN4=false);



void MultiLevelField(Array<double> *lat, CudaRNG *rng_state, Array<complexd> **pp, Array<complexd> **ppfield, int n4, int k4, int n2, int k2, int metrop, int ovrn, int Rmax, bool PrintResultsAtEveryN4=false);


Array<complexd>* MultiLevelTTO(Array<double> *lat, CudaRNG *rng_state, int n4, int k4, int n2, int k2, int metrop, int ovrn, int radius, bool SquaredField, bool alongCharges, bool symmetrize, int perpPoint = 0);

namespace MLgeneric{

//nl0: number of links per time slice at level 0
//nl1: number of links per time slice at level 1
Array<complexd>* MultiLevel(Array<double> *dev_lat, CudaRNG *rng_state, int nl0, int nl1, int n4, int k4, int n2, int k2, int metrop, int ovrn, int Rmax, bool PrintResultsAtEveryN4=false);


//nl0: number of links per time slice at level 0
//nl1: number of links per time slice at level 1
void MultiLevelField(Array<double> *lat, CudaRNG *rng_state, Array<complexd> **pp, Array<complexd> **ppfield, int nl0, int nl1, int n4, int k4, int n2, int k2, int metrop, int ovrn, int Rmax, bool PrintResultsAtEveryN4);
}


namespace ML_TTO_generic{
//nl0: number of links per time slice at level 0
//nl1: number of links per time slice at level 1
Array<complexd>* MultiLevelTTO(Array<double> *lat, CudaRNG *rng_state, int nl0, int nl1, int n4, int k4, int n2, int k2, int metrop, int ovrn, int radius, bool SquaredField, bool alongCharges, bool symmetrize, int perpPoint = 0);
}

}
#endif
