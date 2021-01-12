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

//#include "special_functions.cuh"

#include "parameters.h"
#include "index.h"
#include "tune.h"
#include "plaquette.h"


using namespace std;


namespace U1{


void plaquette(double *lat, double *plaq){
	for(int i = 0; i < 2; ++i) plaq[i] = 0.;
	for(int parity = 0; parity < 2; ++parity){
		#pragma omp parallel for reduction(+:plaq[:2])
		for(int id = 0; id < HalfVolume(); ++id){
			for(int mu = 0; mu < Dirs() - 1; mu++){	
				double tmp = lat[id + parity * HalfVolume() + mu * Volume()];
				int idmu1 = indexEO_neg(id, parity, mu, 1);
				for (int nu = (mu+1); nu < Dirs(); nu++){			
					double plaqi = tmp;
					plaqi += lat[idmu1 + Volume() * nu];
					plaqi -= lat[indexEO_neg(id, parity, nu, 1) + Volume() * mu];
					plaqi -= lat[id + parity * HalfVolume() + nu * Volume()];
					
					plaq[0] += cos(plaqi);
					plaq[1] += sin(plaqi);	
				}
			}
		}
	}
	int numplaqs = 6; //DIRS=4 3D+1
	if(Dirs()==2) numplaqs = 1.;
	else if(Dirs()==3) numplaqs = 3.;
	double norm = 1. / double(Volume() * numplaqs);
	for(int i = 0; i < 2; ++i) plaq[i] *= norm;
}








__global__ void kernel_plaquette(double *lat, complexd *plaq){
    size_t id = threadIdx.x + blockDim.x * blockIdx.x;

	complexd plaq0 = 0.;
	
	if( id < HalfVolume() ){
		for(int parity = 0; parity < 2; ++parity){	
			for(int mu = 0; mu < Dirs() - 1; mu++){	
				double tmp = lat[id + parity * HalfVolume() + mu * Volume()];
				int idmu1 = indexEO_neg(id, parity, mu, 1);
				for (int nu = (mu+1); nu < Dirs(); nu++){			
					double plaqi = tmp;
					plaqi += lat[idmu1 + Volume() * nu];
					plaqi -= lat[indexEO_neg(id, parity, nu, 1) + Volume() * mu];
					plaqi -= lat[id + parity * HalfVolume() + nu * Volume()];
					
					plaq0.real() += cos(plaqi);
					plaq0.imag() += sin(plaqi);	
				}
			}
		}
	}
	reduce_block_1d<complexd>(plaq, plaq0);
}



using namespace U1;

class Plaquette1: Tunable{
private:
	Array<double>* lat;
	complexd plaq;
	complexd *dev_plaq;
	double norm;
	int size;
	double timesec;
#ifdef TIMMINGS
    Timer time;
#endif

   unsigned int sharedBytesPerThread() const { return sizeof(complexd); }
   unsigned int sharedBytesPerBlock(const TuneParam &param) const { return 0; }
   bool tuneSharedBytes() const { return false; } // Don't tune shared memory
   bool tuneGridDim() const { return false; } // Don't tune the grid dimensions.
   unsigned int minThreads() const { return size; }
   void apply(const cudaStream_t &stream){
	TuneParam tp = tuneLaunch(*this, getTuning(), getVerbosity());
	cudaSafeCall(cudaMemset(dev_plaq, 0, sizeof(complexd)));
	kernel_plaquette<<<tp.grid, tp.block, tp.shared_bytes, stream>>>(lat->getPtr(), dev_plaq);
}
public:
   Plaquette1(Array<double>* lat) : lat(lat) {
	size = HalfVolume();
	dev_plaq = (complexd*)dev_malloc(sizeof(complexd));
	
	int numplaqs = 6; //DIRS=4 3D+1
	if(Dirs()==2) numplaqs = 1.;
	else if(Dirs()==3) numplaqs = 3.;
	norm = 1. / double(Volume() * numplaqs);
	timesec = 0.0;  
}
   ~Plaquette1(){ dev_free(dev_plaq);};
   complexd Run(const cudaStream_t &stream){
#ifdef TIMMINGS
    time.start();
#endif
	apply(stream);
	cudaSafeCall(cudaMemcpy(&plaq, dev_plaq, sizeof(complexd), cudaMemcpyDeviceToHost));
	plaq *= norm;
	plaq.real() = 1.-plaq.real();
	//cout << "plaq: " << plaq.real() << '\t' << plaq.imag() << endl;
    CUDA_SAFE_DEVICE_SYNC();
    CUT_CHECK_ERROR("Kernel execution failed");
#ifdef TIMMINGS
	CUDA_SAFE_DEVICE_SYNC( );
    time.stop();
    timesec = time.getElapsedTimeInSec();
#endif
	return plaq;
}
   complexd Run(){	return Run(0);}
   double flops(){	return ((double)flop() * 1.0e-9) / timesec;}
   double bandwidth(){	return (double)bytes() / (timesec * (double)(1 << 30));}
   long long flop() const { return 0;}
   long long bytes() const{ return 0;}
   double time(){	return timesec;}
   void stat(){	cout << "OverRelaxation:  " <<  time() << " s\t"  << bandwidth() << " GB/s\t" << flops() << " GFlops"  << endl;}
  TuneKey tuneKey() const {
    std::stringstream vol, aux;
    vol << PARAMS::Grid[0] << "x";
    vol << PARAMS::Grid[1] << "x";
    vol << PARAMS::Grid[2] << "x";
    vol << PARAMS::Grid[3];
    aux << "threads=" << size;
    return TuneKey(vol.str().c_str(), typeid(*this).name(), aux.str().c_str());
  }
  std::string paramString(const TuneParam &param) const {
    std::stringstream ps;
    ps << "block=(" << param.block.x << "," << param.block.y << "," << param.block.z << ")";
    ps << "shared=" << param.shared_bytes;
    return ps.str();
  }
  void preTune() { }
  void postTune() { }

};








complexd Plaquette(Array<double> *dev_lat, bool print){
	Plaquette1 plaq1(dev_lat);
	complexd plaq = plaq1.Run();
	if(print) cout << "plaq: " << plaq.real() << '\t' << plaq.imag() << endl;
	return plaq;
} 

}
