#ifndef __CHROMOFIELD__
#define __CHROMOFIELD__

#include "random.h"
#include "array.h"

namespace U1{


void CalcChromoFieldPP(Array<complexd> *ploop, Array<complexd> *plaqfield, Array<complexd> *field, int Rmax, int nx, int ny, bool chargeplane);

void CalcChromoFieldWL(Array<complexd> *ploop, Array<complexd> *plaqfield, Array<complexd> *field, int Rmax, int Tmax, int nx, int ny, bool chargeplane);



}
#endif
