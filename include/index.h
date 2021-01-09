#ifndef __INDEX__
#define __INDEX__

#include <cuda.h>

#include "complex.h"

#include "parameters.h"


namespace U1{


inline  __host__   __device__ void indexNO3D(const int id, int x[3]){
	x[2] = (id/(Grid(0) * Grid(1))) % Grid(2);
	x[1] = (id/Grid(0)) % Grid(1);
	x[0] = id % Grid(0);
}

inline  __host__   __device__ complexd exp_ir(double a){
	return complexd(cos(a), sin(a));
}

InlineHostDevice void indexEO(int id, int parity, int x[4]){
	int za = (id / (Grid(0)/2));
	int zb =  (za / Grid(1));
	x[1] = za - zb * Grid(1);
	x[3] = (zb / Grid(2));
	x[2] = zb - x[3] * Grid(2);
	int xodd = (x[1] + x[2] + x[3] + parity) & 1;
	x[0] = (2 * id + xodd )  - za * Grid(0);
 }

InlineHostDevice int indexId(int x[4]){
 return  ((((x[3] * Grid(2) + x[2]) * Grid(1)) + x[1] ) * Grid(0) + x[0]);
}
InlineHostDevice int indexId(int x[4], int parity, int dir){
	int id = ((((x[3] * Grid(2) + x[2]) * Grid(1)) + x[1] ) * Grid(0) + x[0]) >> 1;
 return id + parity * HalfVolume() + Volume() * dir;
}
InlineHostDevice int indexId(int x[4], int dir){
	int id = ((((x[3] * Grid(2) + x[2]) * Grid(1)) + x[1] ) * Grid(0) + x[0]) >> 1;
	int parity = (x[0] + x[1] + x[2] +x[3]) & 1;
 return id + parity * HalfVolume() + Volume() * dir;
}

InlineHostDevice int indexEO_neg(const int id, int parity, int mu, int lmu){
	int x[4];
	indexEO(id, parity, x);
	x[mu] = (x[mu]+lmu+Grid(mu)) % Grid(mu);
	
	int pos = ((((x[3] * Grid(2) + x[2]) * Grid(1)) + x[1] ) * Grid(0) + x[0]) >> 1;
	int oddbit = (x[0] + x[1] + x[2] +x[3]) & 1;
	pos += oddbit  * HalfVolume();
	return pos;
}
InlineHostDevice int indexEO_neg(const int id, int parity, int mu, int lmu, int nu, int lnu){
	int x[4];
	indexEO(id, parity, x);
	x[mu] = (x[mu]+lmu+Grid(mu)) % Grid(mu);
	x[nu] = (x[nu]+lnu+Grid(nu)) % Grid(nu);

	int pos = ((((x[3] * Grid(2) + x[2]) * Grid(1)) + x[1] ) * Grid(0) + x[0]) >> 1;
	int oddbit = (x[0] + x[1] + x[2] +x[3]) & 1;
	pos += oddbit  * HalfVolume();
	return pos;
}


InlineHostDevice int Index_4D_Neig_EO(const int x[], const int dx[], const int X[4]) {
	int y[4];
	for (int i=0; i<4; i++) y[i] = (x[i] + dx[i] + X[i]) % X[i];
	int idx = (((y[3]*X[2] + y[2])*X[1] + y[1])*X[0] + y[0]) >> 1;
	return idx;
}

}
#endif
