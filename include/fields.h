#ifndef __FIELDS_H__
#define __FIELDS_H__

#include "random.h"
#include "array.h"

namespace U1{


void Calc_PPFields(Array<double>* lattice, CudaRNG *rng, bool chargeplane=true, bool multilevel=true);

void Calc_WLFields(Array<double>* lattice, CudaRNG *rng, bool chargeplane=true);







void CalcChromoFieldPP(Array<complexd> *ploop, Array<complexd> *plaqfield, Array<complexd> *field, int Rmin, int Rmax, int nx, int ny, bool chargeplane);

void CalcChromoFieldWL(Array<complexd> *ploop, Array<complexd> *plaqfield, Array<complexd> *field, int Rmin, int Rmax, int Tmax, int nx, int ny, bool chargeplane);



}
#endif
