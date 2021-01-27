#ifndef __PLAQUETTE__
#define __PLAQUETTE__

#include "complex.h"
#include "array.h"


namespace U1{

void plaquette(double *lat, double *plaq);

complexd* Plaquette(Array<double> *dev_lat, complexd *plaq, bool print=false);


class PlaqFieldArg{
	public:
	size_t size;
	complexd* plaqfield;
	complexd* plaq;

	PlaqFieldArg(){
		size = 0;
		plaqfield = 0;
		plaq = 0;
	}
	~PlaqFieldArg(){
		if(plaqfield) dev_free(plaqfield);
		if(plaq) host_free(plaq);
		size = 0;	
	}
};

void PlaquetteFields(Array<double> *lat, PlaqFieldArg* plaqfield, bool spacetime, bool evenoddOrder);
}

#endif
