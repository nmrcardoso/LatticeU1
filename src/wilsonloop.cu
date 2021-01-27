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

		
		
template< class Real>
__global__ void kernel_wilsonloop(Real *lat, complexd *wloop, int Rmax, int Tmax){
    size_t id = threadIdx.x + blockDim.x * blockIdx.x;


	for(int dir = 0; dir < Dirs()-1; dir++)
	for(int r = 0; r < Rmax; r++){
		complexd left = 1.0;
		complexd right = 1.0;
		complexd top = 1.0;
		complexd bottom = 1.0;
		
		int idx = id;
		if(id < Volume())
		for( int ir = 0; ir < r; ++ir ) {
			bottom *= GetValue(lat[idx + dir * Volume()]);
			idx = indexNO_neg(idx, dir, 1);
		}	
		int idl = id;
		int idr = idx;
	
		//Começa aqui a confusão
		for( int t = 0; t < Tmax; ++t ) {

			complexd top = 1.0;
			int idt = indexNO_neg(id, TDir(), t);
			if(id < Volume())
			for( int ir = 0; ir < r; ++ir ) {
				top *= GetValue(lat[idt + dir * Volume()]);
				idt = indexNO_neg(idt, dir, 1);
			}
    
    
    		complexd wl = 0.0;
    		if(id < Volume()) wl = bottom * right * conj(top) * conj(left);
    		
			reduce_block_1d<complexd>(wloop + t + Tmax * r, wl);
    
    
    		
			//actualiza matrizes temporais
			if(id < Volume()){
				left *= GetValue(lat[idl + TDir() * Volume()]);
				right *= GetValue(lat[idr + TDir() * Volume()]);
			}
			idl = indexNO_neg(idl, TDir(), 1);
			idr = indexNO_neg(idr, TDir(), 1);
		}
	}   
}
    
   

template< class Real>
class wilsonloop: Tunable{
private:
	Array<Real>* lat;
	Array<complexd>* wloop;
	Array<complexd>* wloop_dev;
	int R, T;
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
	wloop_dev->Clear();
	kernel_wilsonloop<Real><<<tp.grid, tp.block, tp.shared_bytes, stream>>>(lat->getPtr(), wloop_dev->getPtr(), R, T);
}
public:
   wilsonloop(Array<Real>* lat, int R, int T) : lat(lat), R(R), T(T) {
	size = Volume();
	wloop = new Array<complexd>(Host, R*T);
	wloop_dev = new Array<complexd>(Device, R*T);
	timesec = 0.0;  
}
   ~wilsonloop(){ delete wloop_dev; };
   Array<complexd>* Run(const cudaStream_t &stream){
#ifdef TIMMINGS
    time.start();
#endif
	apply(stream);
	wloop->Copy(wloop_dev);
	for(int i = 0; i < wloop->Size(); i++) wloop->at(i) /= double(Volume()*(Dirs()-1));
    cudaDevSync();
    cudaCheckError("Kernel execution failed");
#ifdef TIMMINGS
	cudaDevSync( );
    time.stop();
    timesec = time.getElapsedTimeInSec();
#endif
	return wloop;
}
   Array<complexd>* Run(){ return Run(0);}
   double flops(){	return ((double)flop() * 1.0e-9) / timesec;}
   double bandwidth(){	return (double)bytes() / (timesec * (double)(1 << 30));}
   long long flop() const { return 0;}
   long long bytes() const{ return 0;}
   double time(){	return timesec;}
   void stat(){	cout << "Wilson loop:  " <<  time() << " s\t"  << bandwidth() << " GB/s\t" << flops() << " GFlops"  << endl;}
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
Array<complexd>* WilsonLoop(Array<Real>* lat, int R, int T){
	Array<Real>* tmp = LatticeConvert(lat, true);
	wilsonloop<Real> wloop(tmp, R, T);
	Array<complexd>* res = wloop.Run();
	delete tmp;
	return res;
}

template Array<complexd>* WilsonLoop(Array<double>* lat, int R, int T);
template Array<complexd>* WilsonLoop(Array<complexd>* lat, int R, int T);

}
