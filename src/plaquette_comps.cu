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


using namespace std;


namespace U1{



template<bool spacetime, bool evenoddOrder>
__global__ void kernel_plaquette_comps(const double *lat, complexd *plaq_comps, complexd* mean_plaq){

    size_t idx = threadIdx.x + blockDim.x * blockIdx.x;
   
	complexd plaq[6];
	for(int d=0; d<6; d++) plaq[d] = 0.0;	
		
	if(evenoddOrder){
		if(spacetime){
			if( idx < Volume() ) {
			   	size_t id = idx;
				int parity = 0;
				if( id >= HalfVolume() ){
					parity = 1;	
					id -= HalfVolume();
				}
				
				SixPlaquette(lat, plaq, id, parity);
				
				int x[4];
				indexEO(id, parity, x);
				int pos = ((((x[3] * Grid(2) + x[2]) * Grid(1)) + x[1] ) * Grid(0) + x[0]);
				for(int d=0; d<6; d++) plaq_comps[pos + d * Volume()] = plaq[d];
			}
		}
		else{	
			if( idx < SpatialVolume() ) {
			   	size_t id = idx;
				int parity = 0;
				if( id >= SpatialVolume()/2 ){
					parity = 1;	
					id -= SpatialVolume()/2;
				}	
				int x[4];
				indexEO(id, parity, x);		
				
				for(x[TDir()] = 0; x[TDir()] < Grid(TDir()); ++x[TDir()]) {				
					int pos = ((((x[3] * Grid(2) + x[2]) * Grid(1)) + x[1] ) * Grid(0) + x[0]) >> 1;
					int oddbit = (x[0] + x[1] + x[2] + x[3]) & 1;			
					SixPlaquette(lat, plaq, pos, oddbit);
				}
				for(int d=0; d<6; d++) plaq[d] /= double(Grid(TDir()));
				
				
				int pos = ((x[2] * Grid(1)) + x[1] ) * Grid(0) + x[0];
				for(int d=0; d<6; d++) plaq_comps[pos + d * SpatialVolume()] = plaq[d];
			}
		}
	}
	else{
		if(spacetime){
			if( idx < Volume() ) {			
				SixPlaquette(lat, plaq, idx);
				for(int d=0; d<6; d++) plaq_comps[idx + d * Volume()] = plaq[d];
			}
		}
		else{	
			if( idx < SpatialVolume() ) {					
				for(int t = 0; t < Grid(TDir()); ++t) {						
					SixPlaquette(lat, plaq, idx + t * SpatialVolume());
				}
				for(int d=0; d<6; d++) plaq[d] /= double(Grid(TDir()));
				for(int d=0; d<6; d++) plaq_comps[idx + d * SpatialVolume()] = plaq[d];
			}
		}
	}
	
	
	for(int d=0; d<6; d++){
		reduce_block_1d<complexd>(mean_plaq + d, plaq[d]);
	  __syncthreads();
	}
}
	
	
template<bool spacetime, bool evenoddOrder>
class PlaqFields: Tunable{
public:
	PlaqFieldArg* fields;
private:
	Array<double>* lat;
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
	cudaSafeCall(cudaMemset(dev_plaq, 0, 6*sizeof(complexd)));
	kernel_plaquette_comps<spacetime, evenoddOrder><<<tp.grid, tp.block, tp.shared_bytes, stream>>>(lat->getPtr(), fields->plaqfield, dev_plaq);
}
public:
   PlaqFields(Array<double>* lat, PlaqFieldArg* fields) : lat(lat), fields(fields) {
   	if(spacetime) size = Volume();
   	else size = SpatialVolume();
	fields->plaqfield = (complexd*)dev_malloc(6*size*sizeof(complexd));
	fields->plaq = (complexd*)safe_malloc(6*sizeof(complexd));
	fields->size = size;
	dev_plaq = (complexd*)dev_malloc(6*sizeof(complexd));
	norm = 1. / double(size);
	timesec = 0.0;  
}
   ~PlaqFields(){ dev_free(dev_plaq);};
   void Run(const cudaStream_t &stream){
#ifdef TIMMINGS
    time.start();
#endif
	apply(stream);
	cudaSafeCall(cudaMemcpy(fields->plaq, dev_plaq, 6*sizeof(complexd), cudaMemcpyDeviceToHost));
	complexd mean = 0.;
	for(int i = 0; i < 6; i++){
		fields->plaq[i] *= norm;
		mean += fields->plaq[i];
		cout << "plaq(" << i << "): " << fields->plaq[i] << endl;
	}
	mean /= 6.0;
	cout << "Mean plaquette: " << mean << endl;
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





template<bool spacetime, bool evenoddOrder>
void PlaquetteFields(Array<double> *lat, PlaqFieldArg* plaqfield){
	PlaqFields<spacetime, evenoddOrder> plaqf(lat, plaqfield);
	plaqf.Run();
}

template<bool spacetime>
void PlaquetteFields(Array<double> *lat, PlaqFieldArg* plaqfield, bool evenoddOrder){
	if(evenoddOrder){
		PlaquetteFields<spacetime, true>(lat, plaqfield);
	}
	else{
		PlaquetteFields<spacetime, false>(lat, plaqfield);
	}
}


void PlaquetteFields(Array<double> *lat, PlaqFieldArg* plaqfield, bool spacetime, bool evenoddOrder){
	if(Dirs() < 4){
		cout << "Only implemented for the 4D case...." << endl;
		Finalize(1);
	}
	if(spacetime){
		PlaquetteFields<true>(lat, plaqfield, evenoddOrder);
	}
	else{
		PlaquetteFields<false>(lat, plaqfield, evenoddOrder);
	}
} 

}
