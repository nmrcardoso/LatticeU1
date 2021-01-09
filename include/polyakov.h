#ifndef __POLYAKOV__
#define __POLYAKOV__

#include "complex.h"


namespace U1{

void polyakov(double *lat, double *poly);

//with tune
complexd dev_polyakov(double *dev_lat);

complexd dev_polyakov(double *dev_lat, complexd *dev_poly, int threads, int blocks);

complexd dev_polyakov2(double *dev_lat, complexd *dev_poly, int radius, int threads, int blocks);

complexd* poly2(double *dev_lat);

complexd* poly2_mhit(double *dev_lat);

//With tune
complexd* Poly2(double *dev_lat, bool multihit);

}

#endif
