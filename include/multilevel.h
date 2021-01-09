#ifndef __MULTILEVEL__
#define __MULTILEVEL__

#include "random.h"

namespace U1{

complexd* MultiLevel(double *dev_lat, cuRNGState *rng_state, int n4, int k4, int n2, int k2, int metrop, int ovrn);

}
#endif
