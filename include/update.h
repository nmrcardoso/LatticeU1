#ifndef __UPDATE__
#define __UPDATE__


#include "random.h"

namespace U1{

void HotStart(double *dev_lat, cuRNGState *rng_state);

void UpdateLattice(double *dev_lat, cuRNGState *rng_state, int metrop, int ovrn);

}

#endif
