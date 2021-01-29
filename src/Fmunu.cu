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
#include "lattice_functions.h"




namespace U1{

using namespace std;


__global__ void kernel_Fmunu(const double *lat, complexd *fmunu_vol, complexd* mean_fmunu){

    size_t idx = threadIdx.x + blockDim.x * blockIdx.x;
   
	complexd fmunu[6];
	for(int d=0; d<6; d++) fmunu[d] = 0.0;
	
	if( idx < Volume() ) {
	   	size_t id = idx;
		int parity = 0;
		if( id >= HalfVolume() ){
			parity = 1;	
			id -= HalfVolume();
		}
		
		Fmunu(lat, fmunu, id, parity);		
		int x[4];
		indexEO(id, parity, x);
		int pos = ((((x[3] * Grid(2) + x[2]) * Grid(1)) + x[1] ) * Grid(0) + x[0]);
		for(int d=0; d<6; d++) fmunu_vol[pos + d * Volume()] = fmunu[d];
	}	
	
	for(int d=0; d<6; d++){
		reduce_block_1d<complexd>(mean_fmunu + d, fmunu[d]);
	  __syncthreads();
	}
}
	
	
	
	
class FmunuClass: Tunable{
private:
	Array<double>* lat;
	Array<complexd> *fmunu_vol;
	Array<complexd> *fmunu;
	Array<complexd> *dev_fmunu;
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
	dev_fmunu->Clear();
	kernel_Fmunu<<<tp.grid, tp.block, tp.shared_bytes, stream>>>(lat->getPtr(), fmunu_vol->getPtr(), dev_fmunu->getPtr());
}
public:
   FmunuClass(Array<double>* lat, Array<complexd> **fmunu_voli, Array<complexd> **fmunui) : lat(lat) {
   	size = Volume();
   	fmunu_vol = new Array<complexd>(Device, 6*size);
   	fmunu = new Array<complexd>(Host, 6);
   	*fmunu_voli = fmunu_vol;
   	*fmunui = fmunu;
   	dev_fmunu = new Array<complexd>(Device, 6);
   	
	norm = 1. / double(size);
	timesec = 0.0;  
}
   ~FmunuClass(){ delete dev_fmunu;};
   void Run(const cudaStream_t &stream){
#ifdef TIMMINGS
    time.start();
#endif
	apply(stream);
	fmunu->Copy(dev_fmunu);
	for(int i = 0; i < 6; i++){
		fmunu->at(i) *= norm;
		cout << "Fmunu(" << i << "): " << fmunu->at(i) << endl;
	}
    cudaDevSync();
    cudaCheckError("Kernel execution failed");
#ifdef TIMMINGS
	cudaDevSync( );
    time.stop();
    timesec = time.getElapsedTimeInSec();
#endif
}
   void Run(){ Run(0); }
   double flops(){	return ((double)flop() * 1.0e-9) / timesec;}
   double bandwidth(){	return (double)bytes() / (timesec * (double)(1 << 30));}
   long long flop() const { return 0;}
   long long bytes() const{ return 0;}
   double get_time(){	return timesec;}
   void stat(){	cout << "Fmunu:  " <<  get_time() << " s\t"  << bandwidth() << " GB/s\t" << flops() << " GFlops"  << endl;}
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




void Fmunu(Array<double> *lat, Array<complexd> **fmunu_vol, Array<complexd> **fmunu){
	FmunuClass cfmunu(lat, fmunu_vol, fmunu);
	cfmunu.Run();
}


}
