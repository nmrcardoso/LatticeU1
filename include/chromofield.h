#ifndef __CHROMOFIELD__
#define __CHROMOFIELD__

#include "random.h"
#include "array.h"

namespace U1{


void CalcChromoField(Array<complexd> *ploop, Array<complexd> *plaqfield, Array<complexd> *field, int radius, int nx, int ny, bool chargeplane);



}
#endif
