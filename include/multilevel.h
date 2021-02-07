#ifndef __MULTILEVEL__
#define __MULTILEVEL__

#include "random.h"
#include "array.h"

namespace U1{




class ML_Fields{
	public:
	Array<complexd>* plaq;
	Array<complexd>* pp;
	Array<complexd>* ppo;
	
	ML_Fields(){
		plaq = 0;
		pp = 0;
		ppo = 0;
	}
	ML_Fields(Array<complexd>* plaq, Array<complexd>* pp, Array<complexd>* ppo) : plaq(plaq), pp(pp), ppo(ppo) {}
	void Set(Array<complexd>* plaq_, Array<complexd>* pp_, Array<complexd>* ppo_){
		plaq = plaq_;
		pp = pp_;
		ppo = ppo_;
	}
	~ML_Fields(){ 
		if(plaq) delete plaq;
		if(pp) delete pp;
		if(ppo) delete ppo;
	}
};




Array<complexd>* MultiLevel(Array<double> *dev_lat, CudaRNG *rng_state, int n4, int k4, int n2, int k2, int metrop, int ovrn, int Rmax, bool PrintResultsAtEveryN4=false);



void MultiLevelField(Array<double> *lat, CudaRNG *rng_state, Array<complexd> **pp, Array<complexd> **ppfield, int n4, int k4, int n2, int k2, int metrop, int ovrn, int Rmax, bool PrintResultsAtEveryN4=false);


ML_Fields* MultiLevelTTO(Array<double> *lat, CudaRNG *rng_state, int n4, int k4, int n2, int k2, int metrop, int ovrn, int radius, bool SquaredField, bool alongCharges, bool symmetrize, int2 perpPoint = make_int2(0,0), bool ppmhit=false, bool plaqmhit=false);

namespace MLgeneric{

//nl0: number of links per time slice at level 0
//nl1: number of links per time slice at level 1
Array<complexd>* MultiLevel(Array<double> *dev_lat, CudaRNG *rng_state, int nl0, int nl1, int n4, int k4, int n2, int k2, int metrop, int ovrn, int Rmax, bool PrintResultsAtEveryN4=false);


//nl0: number of links per time slice at level 0
//nl1: number of links per time slice at level 1
void MultiLevelField(Array<double> *lat, CudaRNG *rng_state, Array<complexd> **pp, Array<complexd> **ppfield, int nl0, int nl1, int n4, int k4, int n2, int k2, int metrop, int ovrn, int Rmax, bool PrintResultsAtEveryN4);
}


namespace ML_TTO_generic{
//nl0: number of links per time slice at level 0
//nl1: number of links per time slice at level 1
ML_Fields* MultiLevelTTO(Array<double> *lat, CudaRNG *rng_state, int nl0, int nl1, int n4, int k4, int n2, int k2, int metrop, int ovrn, int radius, bool SquaredField, bool alongCharges, bool symmetrize, int2 perpPoint = make_int2(0,0), bool ppmhit=false, bool plaqmhit=false);
}

}
#endif
