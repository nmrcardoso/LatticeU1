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

#include "update.h"
#include "staple.h"
#include "enum.h"

#include "tune.h"


namespace U1{

__global__ void kernel_hotstart(double *lat, cuRNGState *rng_state){
    size_t id = threadIdx.x + blockDim.x * blockIdx.x;
    if( id >= HalfVolume() ) return ;

    cuRNGState localState = rng_state[ id ];
	for(int parity = 0; parity < 2; ++parity)
	for(int mu = 0; mu < Dirs(); mu++){
	
		lat[id + parity * HalfVolume() + mu * Volume()] = Random<double>(localState, 0., 2.) * M_PI;
	}
    rng_state[ id ] = localState;
}

void HotStart(Array<double> *dev_lat, CudaRNG *rng_state){
	// kernel number of threads per block and number os blocks
	int threads = 128;
	int blocks = (HalfVolume() + threads - 1) / threads;
	kernel_hotstart<<<blocks,threads>>>(dev_lat->getPtr(), rng_state->getPtr());
}


void metropolis(double *lat){

	std::uniform_real_distribution<double> rand02(0., 2.);
	std::uniform_real_distribution<double> rand01(0,1);

	for(int parity = 0; parity < 2; ++parity)
	for(int mu = 0; mu < Dirs(); mu++){
		#pragma omp parallel for
		for(int id = 0; id < HalfVolume(); ++id){
			double phase_old = lat[id + parity * HalfVolume() + mu * Volume()];
			int idmu1 = indexEO_neg(id, parity, mu, 1);
			double stapleRe = 0., stapleIm = 0.;
			staple(lat, id, parity, mu, stapleRe, stapleIm);			
			double r = std::sqrt( stapleRe*stapleRe + stapleIm*stapleIm );
			double t2 = atan2(stapleIm, stapleRe);

			double new_phase = M_PI * rand02(generator[omp_get_thread_num()]);
			double b = rand01(generator[omp_get_thread_num()]);

			double S1 = cos(phase_old + t2);
			double S2 = cos(new_phase + t2);
			double dS = exp(Beta()*r*(S2-S1));
			if(dS > b){
				lat[id + parity * HalfVolume() + mu * Volume()] = new_phase;
				PARAMS::accept_ratio += 1.;
			}
		}
	}
}




void overrelaxation(double *lat){
	for(int parity = 0; parity < 2; ++parity)
	for(int mu = 0; mu < Dirs(); mu++){
		#pragma omp parallel for
		for(int id = 0; id < HalfVolume(); ++id){
			double stapleRe = 0., stapleIm = 0.;
			staple(lat, id, parity, mu, stapleRe, stapleIm);
			int pos = id + parity * HalfVolume() + mu * Volume();
			double phase_old = lat[pos];
			double t2 = atan2(stapleIm, stapleRe);
			double new_phase = fmod(6.* M_PI - phase_old - 2. * t2, 2.* M_PI);
			lat[pos] = new_phase;
		}
	}
}










__global__ void kernel_metropolis(double *lat, int parity, int mu, cuRNGState *rng_state){
    size_t id = threadIdx.x + blockDim.x * blockIdx.x;
    if( id >= HalfVolume() ) return ;
    cuRNGState localState = rng_state[ id ];

	double phase_old = lat[id + parity * HalfVolume() + mu * Volume()];
	int idmu1 = indexEO_neg(id, parity, mu, 1);
	double stapleRe = 0., stapleIm = 0.;
	staple(lat, id, parity, mu, stapleRe, stapleIm);			
	double r = sqrt( stapleRe*stapleRe + stapleIm*stapleIm );
	double t2 = atan2(stapleIm, stapleRe);

	double new_phase = Random<double>(localState) * 2. * M_PI;
	double b = Random<double>(localState);

	double S1 = cos(phase_old + t2);
	double S2 = cos(new_phase + t2);
	double dS = exp(Beta()*r*(S2-S1));
	if(dS > b){
		lat[id + parity * HalfVolume() + mu * Volume()] = new_phase;
	}	
    rng_state[ id ] = localState;
}


__global__ void kernel_overrelaxation(double *lat, int parity, int mu){
    size_t id = threadIdx.x + blockDim.x * blockIdx.x;
    if( id >= HalfVolume() ) return ;
	double stapleRe = 0., stapleIm = 0.;
	staple(lat, id, parity, mu, stapleRe, stapleIm);
	int pos = id + parity * HalfVolume() + mu * Volume();
	double phase_old = lat[pos];
	double t2 = atan2(stapleIm, stapleRe);
	double new_phase = fmod(6.* M_PI - phase_old - 2. * t2, 2.* M_PI);
	lat[pos] = new_phase;
}






void UpdateLattice1(Array<double> *dev_lat, CudaRNG *rng_state, int metrop, int ovrn){
	int threads = 128;
	int blocks = (HalfVolume() + threads - 1) / threads;
	// metropolis algorithm
	for(int m = 0; m < metrop; ++m)
	for(int parity = 0; parity < 2; ++parity)
	for(int mu = 0; mu < Dirs(); ++mu)
		kernel_metropolis<<<blocks,threads>>>(dev_lat->getPtr(), parity, mu, rng_state->getPtr());
	// overrelaxation algorithm 
	for(int ovr = 0; ovr < ovrn; ++ovr)
	for(int parity = 0; parity < 2; ++parity)
	for(int mu = 0; mu < Dirs(); ++mu)
		kernel_overrelaxation<<<blocks,threads>>>(dev_lat->getPtr(), parity, mu);	
}





















using namespace U1;

class Metropolis: Tunable{
private:
	Array<double>* lat;
	CudaRNG *rng_state;
	int metrop;
	int parity;
	int mu;
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
	kernel_metropolis<<<tp.grid,tp.block, 0, stream>>>(lat->getPtr(), parity, mu, rng_state->getPtr());
}
public:
   Metropolis(Array<double>* lat, CudaRNG *rng_state, int metrop) : lat(lat), rng_state(rng_state), metrop(metrop){
	size = HalfVolume();
	timesec = 0.0;  
}
   ~Metropolis(){};
   void Run(const cudaStream_t &stream){
#ifdef TIMMINGS
    time.start();
#endif
	for(int m = 0; m < metrop; ++m)
	for(parity = 0; parity < 2; ++parity)
	for(mu = 0; mu < Dirs(); ++mu)
	    apply(stream);
    CUDA_SAFE_DEVICE_SYNC();
    CUT_CHECK_ERROR("Kernel execution failed");
#ifdef TIMMINGS
	CUDA_SAFE_DEVICE_SYNC( );
    time.stop();
    timesec = time.getElapsedTimeInSec();
#endif
}
   void Run(){	return Run(0);}
   double flops(){	return ((double)flop() * 1.0e-9) / timesec;}
   double bandwidth(){	return (double)bytes() / (timesec * (double)(1 << 30));}
   long long flop() const { return 0;}
   long long bytes() const{ return 0;}
   double time(){	return timesec;}
   void stat(){	cout << "Metropolis:  " <<  time() << " s\t"  << bandwidth() << " GB/s\t" << flops() << " GFlops"  << endl;}
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
  void preTune() {
  	lat->Backup();
	rng_state->Backup();
  }
  void postTune() {  
	lat->Restore();
	rng_state->Restore();
 }

};



class OverRelaxation: Tunable{
private:
	Array<double>* lat;
	int ovrn;
	int parity;
	int mu;
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
	kernel_overrelaxation<<<tp.grid,tp.block, 0, stream>>>(lat->getPtr(), parity, mu);
}
public:
   OverRelaxation(Array<double>* lat, int ovrn) : lat(lat), ovrn(ovrn){
	size = HalfVolume();
	timesec = 0.0;  
}
   ~OverRelaxation(){};
   void Run(const cudaStream_t &stream){
#ifdef TIMMINGS
    time.start();
#endif
	for(int m = 0; m < ovrn; ++m)
	for(parity = 0; parity < 2; ++parity)
	for(mu = 0; mu < Dirs(); ++mu)
	    apply(stream);
    CUDA_SAFE_DEVICE_SYNC();
    CUT_CHECK_ERROR("Kernel execution failed");
#ifdef TIMMINGS
	CUDA_SAFE_DEVICE_SYNC( );
    time.stop();
    timesec = time.getElapsedTimeInSec();
#endif
}
   void Run(){	return Run(0);}
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
  void preTune() {
	lat->Backup();		
  }
  void postTune() {  
	lat->Restore();
 }

};
























void UpdateLattice(Array<double> *dev_lat, CudaRNG *rng_state, int metrop, int ovrn){
	// metropolis algorithm
	Metropolis mtp(dev_lat, rng_state, metrop);
	mtp.Run();
	// overrelaxation algorithm
	OverRelaxation ovr(dev_lat, ovrn);
	ovr.Run();
}

}
