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
#include "array.h"


using namespace std;


namespace U1{


template<bool EO_TO_NO_ORDER>
__global__ void kernel_convert_EO_NO(const double *in, double *out){
	size_t id = threadIdx.x + blockDim.x * blockIdx.x;
	if( id >= Volume() ) return;
	if(EO_TO_NO_ORDER){
		int parity = 0;
		if( id >= HalfVolume() ){
			parity = 1;	
			id -= HalfVolume();
		}
		int x[4];
		indexEO(id, parity, x);
		
		size_t idx = indexId(x);
		for(int dir = 0; dir < Dirs(); dir++){
			out[idx + dir * Volume()] = in[id + parity * HalfVolume() + dir * Volume()];
		}
	}
	else{
		int x[4];
		indexNO(id, x);
		
		size_t idx = indexId(x) >> 1;
		int parity = GetParity(x);
		for(int dir = 0; dir < Dirs(); dir++){
			out[id + parity * HalfVolume() + dir * Volume()] = in[idx + dir * Volume()];
		}



	}
}


template<bool EO_TO_NO_ORDER>
class ConvLattice_EO_NO: Tunable{
public:
private:
	Array<double>* lat;
	Array<double>* latno;
	int size;
	double timesec;
#ifdef TIMMINGS
    Timer time;
#endif

   unsigned int sharedBytesPerThread() const { return 0; }
   unsigned int sharedBytesPerBlock(const TuneParam &param) const { return 0; }
   bool tuneSharedBytes() const { return false; } // Don't tune shared memory
   bool tuneGridDim() const { return false; } // Don't tune the grid dimensions.
   unsigned int minThreads() const { return size; }
   void apply(const cudaStream_t &stream){
	TuneParam tp = tuneLaunch(*this, getTuning(), getVerbosity());
	kernel_convert_EO_NO<EO_TO_NO_ORDER><<<tp.grid, tp.block, 0, stream>>>(lat->getPtr(), latno->getPtr());
}
public:
   ConvLattice_EO_NO(Array<double>* lat) : lat(lat) {
   	size = Volume();
	latno = new Array<double>(Device, Dirs()*size);
	timesec = 0.0;  
}
   ~ConvLattice_EO_NO(){ };
   Array<double>* Run(const cudaStream_t &stream){
#ifdef TIMMINGS
    time.start();
#endif
	apply(stream);
    cudaDevSync();
    cudaCheckError("Kernel execution failed");
#ifdef TIMMINGS
	cudaDevSync( );
    time.stop();
    timesec = time.getElapsedTimeInSec();
#endif
	return latno;
}
   Array<double>* Run(){ return Run(0); }
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


Array<double>* LatticeConvert(Array<double>* lat, bool eo_to_no){
	if(eo_to_no){
		ConvLattice_EO_NO<true> cv(lat);
		return cv.Run();
	}
	else{
		ConvLattice_EO_NO<false> cv(lat);
		return cv.Run();	
	}
}







}
