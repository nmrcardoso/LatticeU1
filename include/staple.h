#ifndef __STAPLE__
#define __STAPLE__




#include <iostream>
#include <math.h> 
#include <time.h> 
#include <random>
#include <vector> 
#include <fstream>
#include <omp.h>

#include <cuda.h>
#include <curand_kernel.h>

#include "timer.h"
#include "cuda_error_check.h"
#include "alloc.h"
#include "reduce_block_1d.h"
#include "complex.h"



#include "parameters.h"
#include "index.h"


namespace U1{

InlineHostDevice void staple_old(const double *lat, const int id, const int parity, const int mu, double &stapleRe, double &stapleIm){
	stapleRe = 0., stapleIm = 0.;	
	int idmu1 = indexEO_neg(id, parity, mu, 1);			
	for(int nu = 0; nu < Dirs(); nu++)  if(mu != nu) {
		double upperStaple = lat[idmu1 + Volume() * nu];
		upperStaple -= lat[indexEO_neg(id, parity, nu, 1) + Volume() * mu];
		upperStaple -= lat[id + parity * HalfVolume() + nu * Volume()];
		
		double lowerStaple = -lat[indexEO_neg(id, parity, mu, 1, nu, -1) + Volume() * nu];	
		lowerStaple -= lat[indexEO_neg(id, parity, nu, -1) + Volume() * mu];	
		lowerStaple += lat[indexEO_neg(id, parity, nu, -1) + Volume() * nu];	
		
		stapleRe += cos(upperStaple) + cos(lowerStaple);
		stapleIm += sin(upperStaple) + sin(lowerStaple);
	}
}

InlineHostDevice void staple(const double *lat, const int id, const int parity, const int mu, double &stapleRe, double &stapleIm){
	
	double stapleReSS = 0., stapleImSS = 0.;
	
	double stapleReST = 0., stapleImST = 0.;
	int idmu1 = indexEO_neg(id, parity, mu, 1);			
	for(int nu = 0; nu < Dirs(); nu++)  if(mu != nu) {
		double upperStaple = lat[idmu1 + Volume() * nu];
		upperStaple -= lat[indexEO_neg(id, parity, nu, 1) + Volume() * mu];
		upperStaple -= lat[id + parity * HalfVolume() + nu * Volume()];
		
		double lowerStaple = -lat[indexEO_neg(id, parity, mu, 1, nu, -1) + Volume() * nu];	
		lowerStaple -= lat[indexEO_neg(id, parity, nu, -1) + Volume() * mu];	
		lowerStaple += lat[indexEO_neg(id, parity, nu, -1) + Volume() * nu];	
		
		if( mu == TDir() || nu == TDir() ){
			stapleReST += cos(upperStaple) + cos(lowerStaple);
			stapleImST += sin(upperStaple) + sin(lowerStaple);
		
		}
		else{
			stapleReSS += cos(upperStaple) + cos(lowerStaple);
			stapleImSS += sin(upperStaple) + sin(lowerStaple);
		
		}
	}
	stapleRe = stapleReSS / Aniso() + stapleReST * Aniso();
	stapleIm = stapleImSS / Aniso() + stapleImST * Aniso();	
}

InlineHostDevice complexd Staple(const double *lat, const int id, const int parity, const int mu){

	complexd stapleSS = 0.0;
	complexd stapleST = 0.0;
	
	int idmu1 = indexEO_neg(id, parity, mu, 1);			
	for(int nu = 0; nu < Dirs(); nu++)  if(mu != nu) {
		double upperStaple = lat[idmu1 + Volume() * nu];
		upperStaple -= lat[indexEO_neg(id, parity, nu, 1) + Volume() * mu];
		upperStaple -= lat[id + parity * HalfVolume() + nu * Volume()];
		
		double lowerStaple = -lat[indexEO_neg(id, parity, mu, 1, nu, -1) + Volume() * nu];	
		lowerStaple -= lat[indexEO_neg(id, parity, nu, -1) + Volume() * mu];	
		lowerStaple += lat[indexEO_neg(id, parity, nu, -1) + Volume() * nu];	
		
		if( mu == TDir() || nu == TDir() ){
			stapleST += exp_ir(upperStaple) + exp_ir(lowerStaple);		
		}
		else{
			stapleSS += exp_ir(upperStaple) + exp_ir(lowerStaple);		
		}
	}
	return stapleSS / Aniso() + stapleST * Aniso();
}





InlineHostDevice complexd Staple(const complexd *lat, const int id, const int parity, const int mu){

	complexd stapleSS = 0.0;
	complexd stapleST = 0.0;
	
	int idmu1 = indexEO_neg(id, parity, mu, 1);			
	for(int nu = 0; nu < Dirs(); nu++)  if(mu != nu) {
		complexd upperStaple = lat[idmu1 + Volume() * nu];
		upperStaple *= conj(lat[indexEO_neg(id, parity, nu, 1) + Volume() * mu]);
		upperStaple *= conj(lat[id + parity * HalfVolume() + nu * Volume()]);
		
		complexd lowerStaple = conj(lat[indexEO_neg(id, parity, mu, 1, nu, -1) + Volume() * nu]);	
		lowerStaple *= conj(lat[indexEO_neg(id, parity, nu, -1) + Volume() * mu]);	
		lowerStaple *= conj(lat[indexEO_neg(id, parity, nu, -1) + Volume() * nu]);	
		
		if( mu == TDir() || nu == TDir() ){
			stapleST += upperStaple + lowerStaple;		
		}
		else{
			stapleSS += upperStaple + lowerStaple;		
		}
	}
	
	return stapleSS / Aniso() + stapleST * Aniso();
}



















InlineHostDevice void Staple(const double *lat, const int id, const int parity, const int mu, complexd &stapleSS, complexd &stapleST){
	
	stapleSS = 0.0;
	stapleST = 0.0;
	
	int idmu1 = indexEO_neg(id, parity, mu, 1);			
	for(int nu = 0; nu < Dirs(); nu++)  if(mu != nu) {
		double upperStaple = lat[idmu1 + Volume() * nu];
		upperStaple -= lat[indexEO_neg(id, parity, nu, 1) + Volume() * mu];
		upperStaple -= lat[id + parity * HalfVolume() + nu * Volume()];
		
		double lowerStaple = -lat[indexEO_neg(id, parity, mu, 1, nu, -1) + Volume() * nu];	
		lowerStaple -= lat[indexEO_neg(id, parity, nu, -1) + Volume() * mu];	
		lowerStaple += lat[indexEO_neg(id, parity, nu, -1) + Volume() * nu];	
		
		if( mu == TDir() || nu == TDir() ){
			stapleST += exp_ir(upperStaple) + exp_ir(lowerStaple);		
		}
		else{
			stapleSS += exp_ir(upperStaple) + exp_ir(lowerStaple);		
		}
	}
}












}
#endif
