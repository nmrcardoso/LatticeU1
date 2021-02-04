#ifndef __LATTICE_FUNCTIONS__
#define __LATTICE_FUNCTIONS__




#include <iostream>
#include <math.h> 
#include <time.h> 
#include <random>
#include <vector> 
#include <fstream>
#include <omp.h>

#include <cuda.h>
#include <curand_kernel.h>

#include "complex.h"
#include "parameters.h"
#include "index.h"
#include "staple.h"


namespace U1{


inline __device__ __host__ complexd Fmunu(const double *lat, const int id, const int parity, const int mu, const int nu) {
	complexd F = 0.0;
	double plaq = lat[id + parity * HalfVolume() + mu * Volume()];
	int idmu1 = indexEO_neg(id, parity, mu, 1);
	plaq += lat[idmu1 + Volume() * nu];
	plaq -= lat[indexEO_neg(id, parity, nu, 1) + Volume() * mu];
	plaq -= lat[id + parity * HalfVolume() + nu * Volume()];
	F += exp_ir(plaq);
	
	plaq = lat[id + parity * HalfVolume() + nu * Volume()];
	idmu1 = indexEO_neg(id, parity, nu, 1, mu, -1);
	plaq -= lat[idmu1 + Volume() * mu];
	idmu1 = indexEO_neg(id, parity, mu, -1);
	plaq -= lat[idmu1 + Volume() * nu];
	plaq += lat[idmu1 + mu * Volume()];
	F += exp_ir(plaq);	
	
	idmu1 = indexEO_neg(id, parity, mu, -1);
	plaq = -lat[idmu1 + mu * Volume()];
	idmu1 = indexEO_neg(id, parity, nu, -1, mu, -1);
	plaq -= lat[idmu1 + Volume() * nu];
	plaq += lat[idmu1 + Volume() * mu];
	idmu1 = indexEO_neg(id, parity, nu, -1);
	plaq += lat[idmu1 + Volume() * nu];
	F += exp_ir(plaq);
	
	//idmu1 = indexEO_neg(id, parity, nu, -1);
	plaq = -lat[idmu1 + Volume() * nu];
	plaq += lat[idmu1 + Volume() * mu];
	idmu1 = indexEO_neg(id, parity, nu, -1, mu, 1);
	plaq += lat[idmu1 + Volume() * nu];
	plaq -= lat[id + parity * HalfVolume() + mu * Volume()];
	F += exp_ir(plaq);
	
	F -= conj(F);
	F *= 0.125;	
	
	return F;
}

inline   __device__ __host__ void Fmunu(const double *lat, complexd *fmunu, const int id, const int parity){
  fmunu[0] += Fmunu( lat, id, parity, 0, 3 );
  fmunu[1] += Fmunu( lat, id, parity, 1, 3 );
  fmunu[2] += Fmunu( lat, id, parity, 2, 3 );
  fmunu[3] += Fmunu( lat, id, parity, 1, 2 );
  fmunu[4] += Fmunu( lat, id, parity, 2, 0 );
  fmunu[5] += Fmunu( lat, id, parity, 0, 1 );
}





inline __device__ __host__ complexd PlaquetteF(const double *lat, const int id, const int parity, const int mu, const int nu) {
	double plaq = lat[id + parity * HalfVolume() + mu * Volume()];
	int idmu1 = indexEO_neg(id, parity, mu, 1);
	plaq += lat[idmu1 + Volume() * nu];
	plaq -= lat[indexEO_neg(id, parity, nu, 1) + Volume() * mu];
	plaq -= lat[id + parity * HalfVolume() + nu * Volume()];
	return 1.0 - exp_ir(plaq);
}

inline   __device__ __host__ void SixPlaquette(const double *lat, complexd *plaq, const int id, const int parity){
  plaq[0] += PlaquetteF( lat, id, parity, 0, 3 );
  plaq[1] += PlaquetteF( lat, id, parity, 1, 3 );
  plaq[2] += PlaquetteF( lat, id, parity, 2, 3 );
  plaq[3] += PlaquetteF( lat, id, parity, 1, 2 );
  plaq[4] += PlaquetteF( lat, id, parity, 2, 0 );
  plaq[5] += PlaquetteF( lat, id, parity, 0, 1 );
}


inline __device__ __host__ complexd PlaquetteF(const double *lat, const int id, const int mu, const int nu) {
	double plaq = lat[id + mu * Volume()];
	plaq += lat[indexNO_neg(id, mu, 1) + Volume() * nu];
	plaq -= lat[indexNO_neg(id, nu, 1) + Volume() * mu];
	plaq -= lat[id + nu * Volume()];
	return 1.0 - exp_ir(plaq);
}

inline   __device__ __host__ void SixPlaquette(const double *lat, complexd *plaq, const int id){
  plaq[0] += PlaquetteF( lat, id, 0, 3 );
  plaq[1] += PlaquetteF( lat, id, 1, 3 );
  plaq[2] += PlaquetteF( lat, id, 2, 3 );
  plaq[3] += PlaquetteF( lat, id, 1, 2 );
  plaq[4] += PlaquetteF( lat, id, 2, 0 );
  plaq[5] += PlaquetteF( lat, id, 0, 1 );
}




InlineHostDevice double MetropolisFunc(double *lat, const int id, const int parity, const int mu, double new_phase){
	double phase_old = lat[id + parity * HalfVolume() + mu * Volume()];
	complexd staple = Staple(lat, id, parity, mu);			
	double r = staple.abs();
	double t2 = atan2(staple.imag(), staple.real());

	double S1 = cos(phase_old + t2);
	double S2 = cos(new_phase + t2);
	double dS = exp(Beta()*r*(S2-S1));
	
	return dS;
}


InlineHostDevice double OvrFunc(double *lat, const int id, const int parity, const int mu){
	complexd staple = Staple(lat, id, parity, mu);
	double phase_old = lat[id + parity * HalfVolume() + mu * Volume()];
	double t2 = atan2(staple.imag(), staple.real());
	double new_phase = fmod(6.* M_PI - phase_old - 2. * t2, 2.* M_PI);
	return new_phase;
}


InlineHostDevice complexd MultiHit(const double *lat, const int id, const int parity, const int mu){
	complexd staple = Staple(lat, id, parity, mu);				
	double alpha = staple.abs();
	double ba = Beta() * alpha;
	double temp = cyl_bessel_i1(ba)/(cyl_bessel_i0(ba)*alpha);
	//double temp = besseli1(ba)/(besseli0(ba)*alpha);
	complexd val(temp * staple.real(), -temp * staple.imag());
	return val;
}

InlineHostDevice complexd MultiHit(const complexd *lat, const int id, const int parity, const int mu){
	complexd staple = Staple(lat, id, parity, mu);				
	double alpha = staple.abs();
	double ba = Beta() * alpha;
	double temp = cyl_bessel_i1(ba)/(cyl_bessel_i0(ba)*alpha);
	//double temp = besseli1(ba)/(besseli0(ba)*alpha);
	complexd val(temp * staple.real(), -temp * staple.imag());
	return val;
}






}
#endif
