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
#include "tune.h"
#include "array.h"


using namespace std;


namespace U1{


template<class Real>
__global__ void kernel_Test_Link(const Real *in, complexd *out){
	uint id = threadIdx.x + blockDim.x * blockIdx.x;
	complexd link = 0.0;
	if( id < Volume()  ) {	
		for(int dir = 0; dir < Dirs(); dir++){
			complexd tmp = GetValue<Real>(in[id + dir * Volume()]);
			link += tmp * conj(tmp);
		}
	}
	reduce_block_1d<complexd>(out, link);
}


template<class Real>
class LinkTestSum: Tunable{
public:
private:
	Array<Real>* lat;
	complexd *dev_res;
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
	cudaSafeCall(cudaMemset(dev_res, 0, sizeof(complexd)));
	kernel_Test_Link<Real><<<tp.grid, tp.block, tp.shared_bytes, stream>>>(lat->getPtr(), dev_res);
}
public:
   LinkTestSum(Array<Real>* lat) : lat(lat) {
   	size = Volume();
   	dev_res = (complexd*) dev_malloc(sizeof(complexd));
	timesec = 0.0;  
}
   ~LinkTestSum(){dev_free(dev_res); };
   complexd Run(const cudaStream_t &stream){
#ifdef TIMMINGS
    time.start();
#endif
	apply(stream);
	complexd res;
	cudaSafeCall(cudaMemcpy(&res, dev_res, sizeof(complexd), cudaMemcpyDeviceToHost));
	res /= double(Volume()*Dirs());
    cudaDevSync();
    cudaCheckError("Kernel execution failed");
#ifdef TIMMINGS
	cudaDevSync( );
    time.stop();
    timesec = time.getElapsedTimeInSec();
#endif
	return res;
}
   complexd Run(){ return Run(0); }
   double flops(){	return ((double)flop() * 1.0e-9) / timesec;}
   double bandwidth(){	return (double)bytes() / (timesec * (double)(1 << 30));}
   long long flop() const { return 0;}
   long long bytes() const{ return 0;}
   double get_time(){	return timesec;}
   void stat(){	cout << "Link Test:  " <<  get_time() << " s\t"  << bandwidth() << " GB/s\t" << flops() << " GFlops"  << endl;}
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

template<class Real>
complexd TestLink(Array<Real>* lat){
	LinkTestSum<Real> calc(lat);
	return calc.Run();
}

template complexd TestLink<double>(Array<double>* lat);
template complexd TestLink<complexd>(Array<complexd>* lat);






}
