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
#include "staple.h"
#include "random.h"


#include "tune.h"

using namespace std;

namespace U1{

template<int multilevel>
__global__ void kernel_metropolis_multilevel(double *lat, int parity, int mu, cuRNGState *rng_state){
    size_t id = threadIdx.x + blockDim.x * blockIdx.x;
    if( id >= HalfVolume() ) return ;
    
	int x[4];
	indexEO(id, parity, x);
	if((x[TDir()]%multilevel) || mu==3){	
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
}



template<int multilevel>
class Metropolis_ML: Tunable{
private:
	double* lat;
	double* tmp;
	cuRNGState *rng_state;
	cuRNGState *state;
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
	kernel_metropolis_multilevel<multilevel><<<tp.grid, tp.block, 0, stream>>>(lat, parity, mu, rng_state);
}
public:
   Metropolis_ML(double* lat, cuRNGState *rng_state, int metrop) : lat(lat), rng_state(rng_state), metrop(metrop){
	size = HalfVolume();
	timesec = 0.0;  
}
   ~Metropolis_ML(){};
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
	tmp = (double*)dev_malloc(Volume()*Dirs()*sizeof(double));
	cudaSafeCall(cudaMemcpy(tmp, lat, Volume()*Dirs()*sizeof(double), cudaMemcpyDeviceToDevice));
	
	state = (cuRNGState*)dev_malloc(HalfVolume()*sizeof(cuRNGState));	
	cudaSafeCall(cudaMemcpy(state, rng_state, HalfVolume()*sizeof(cuRNGState), cudaMemcpyDeviceToDevice));	
  }
  void postTune() {  
	cudaSafeCall(cudaMemcpy(lat, tmp, Volume()*Dirs()*sizeof(double), cudaMemcpyDeviceToDevice));
	cudaSafeCall(cudaMemcpy(rng_state, state, HalfVolume()*sizeof(cuRNGState), cudaMemcpyDeviceToDevice));
	dev_free(tmp);
	dev_free(state);
 }

};


template<int multilevel>
__global__ void kernel_overrelaxation_multilevel(double *lat, int parity, int mu){
    size_t id = threadIdx.x + blockDim.x * blockIdx.x;
    if( id >= HalfVolume() ) return ;
    
	int x[4];
	indexEO(id, parity, x);
	if((x[TDir()]%multilevel) || mu==3){
		double stapleRe = 0., stapleIm = 0.;
		staple(lat, id, parity, mu, stapleRe, stapleIm);
		int pos = id + parity * HalfVolume() + mu * Volume();
		double phase_old = lat[pos];
		double t2 = atan2(stapleIm, stapleRe);
		double new_phase = fmod(6.* M_PI - phase_old - 2. * t2, 2.* M_PI);
		lat[pos] = new_phase;
	}
}


template<int multilevel>
class OverRelaxation_ML: Tunable{
private:
	double* lat;
	double* tmp;
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
	kernel_overrelaxation_multilevel<multilevel><<<tp.grid, tp.block, 0, stream>>>(lat, parity, mu);
}
public:
   OverRelaxation_ML(double* lat, int ovrn) : lat(lat), ovrn(ovrn){
	size = HalfVolume();
	timesec = 0.0;  
}
   ~OverRelaxation_ML(){};
   void Run(const cudaStream_t &stream){
#ifdef TIMMINGS
    time.start();
#endif
	for(int m = 0; m < ovrn; ++m)
	for(parity = 0; parity < 2; ++parity)
	for(mu = 0; mu < Dirs(); ++mu){
		//cout << multilevel << '\t' << mu << '\t' << parity << '\t' << m << endl;
	    apply(stream);
    }
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
	tmp = (double*)dev_malloc(Volume()*Dirs()*sizeof(double));
	cudaSafeCall(cudaMemcpy(tmp, lat, Volume()*Dirs()*sizeof(double), cudaMemcpyDeviceToDevice));		
  }
  void postTune() {  
	cudaSafeCall(cudaMemcpy(lat, tmp, Volume()*Dirs()*sizeof(double), cudaMemcpyDeviceToDevice));
	dev_free(tmp);
 }

};
















template<bool multihit>
__global__ void kernel_l2_multilevel_0(double *lat, complexd *poly){
    size_t id = threadIdx.x + blockDim.x * blockIdx.x;   
	if( id >= Volume() ) return;
	int parity = 0;
	if( id >= Volume()/2 ){
		parity = 1;	
		id -= Volume()/2;
	}	
	int x[4];
	indexEO(id, parity, x);
	int id3d = ((((x[3] * Grid(2) + x[2]) * Grid(1)) + x[1] ) * Grid(0) + x[0]);
	
	if(multihit){
		double W_re, W_im;	
		staple(lat, id, parity, TDir(), W_re, W_im);			
		double alpha = sqrt(W_re*W_re+W_im*W_im);
		double ba = Beta() * alpha;
		double temp = cyl_bessel_i1(ba)/(cyl_bessel_i0(ba)*alpha);
		//double temp = besseli1(ba)/(besseli0(ba)*alpha);
		complexd val(temp*W_re, -temp*W_im);
		poly[id3d] = val;
	}
	else{
		poly[id3d] = exp_ir(lat[ indexId(x, TDir()) ]);
	}
}


template< bool multihit>
class Polyakov_Volume: Tunable{
private:
	double* lat;
	complexd* poly;
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
	kernel_l2_multilevel_0<multihit><<<tp.grid, tp.block, 0, stream>>>(lat, poly);
}
public:
   Polyakov_Volume(double* lat) : lat(lat) {
	size = Volume();
	poly = (complexd*)dev_malloc(Volume()*sizeof(complexd));
	timesec = 0.0;  
}
   ~Polyakov_Volume(){ };
   complexd* Run(const cudaStream_t &stream){
#ifdef TIMMINGS
    time.start();
#endif
	apply(stream);
    CUDA_SAFE_DEVICE_SYNC();
    CUT_CHECK_ERROR("Kernel execution failed");
#ifdef TIMMINGS
	CUDA_SAFE_DEVICE_SYNC( );
    time.stop();
    timesec = time.getElapsedTimeInSec();
#endif
	return poly;
}
   complexd* Run(){	return Run(0);}
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



__global__ void kernel_l2_multilevel_1(complexd *poly, complexd *l2, int radius){
    size_t id = threadIdx.x + blockDim.x * blockIdx.x;    
	if(id >= SpatialVolume()) return;		
	int x[3];
	indexNO3D(id, x);
	
	int nlayers = Grid(TDir())/2;
	for(int r = 1; r <= radius; ++r){	
		for(int dir = 0; dir < TDir(); dir++){		
			int layer = 0;
			for(int t = 0; t < Grid(TDir()); t+=2){
				complexd pl0 = 1.;
				complexd pl1 = 1.;
				for(x[TDir()] = t; x[TDir()] < t+2; ++x[TDir()]){
					pl0 *= (poly[(((x[3] * Grid(2) + x[2]) * Grid(1)) + x[1] ) * Grid(0) + x[0]]);
					int xold = x[dir];
					x[dir] = (x[dir] + r) % Grid(dir);
					pl1 *= conj(poly[(((x[3] * Grid(2) + x[2]) * Grid(1)) + x[1] ) * Grid(0) + x[0]]);
									
					x[dir] = xold;
				}			
				complexd pl= pl0 * pl1;
				//int pos = id + SpatialVolume() * layer + nlayers * SpatialVolume() * (r-1) + nlayers * SpatialVolume() * radius * dir;			
				int pos = id + SpatialVolume() * (r-1) + SpatialVolume() * radius * dir + SpatialVolume() * radius * (Dirs()-1) * layer;
				l2[pos] = pl + l2[pos];
				layer++;
			}
		}
	}
}



class L2ML: Tunable{
private:
	complexd* poly;
	complexd *l2;
	complexd *temp_l2;
	size_t sl2;
	int radius;
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
	kernel_l2_multilevel_1<<<tp.grid, tp.block, tp.shared_bytes, stream>>>(poly, l2, radius);
}
public:	
   L2ML(complexd *poly, complexd *l2, size_t sl2, int radius) : poly(poly), l2(l2), sl2(sl2), radius(radius) {
	size = SpatialVolume();
	timesec = 0.0;  
}
   ~L2ML(){ };
   void Run(const cudaStream_t &stream){
#ifdef TIMMINGS
    time.start();
#endif
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
	temp_l2 = (complexd*)dev_malloc(sl2*sizeof(complexd));	
	cudaSafeCall(cudaMemcpy(temp_l2, l2, sl2*sizeof(complexd), cudaMemcpyDeviceToDevice));	
  }
  void postTune() {  
	cudaSafeCall(cudaMemcpy(l2, temp_l2, sl2*sizeof(complexd), cudaMemcpyDeviceToDevice));
	dev_free(temp_l2);
 }

};





__global__ void kernel_l2avg_l4_multilevel(complexd *dev_l2, complexd *dev_l4, int radius, double l2norm){
    size_t id = threadIdx.x + blockDim.x * blockIdx.x;
    size_t size = SpatialVolume() * radius * (Dirs()-1);
    if(id >= size) return;			
	
	int nl2 = Grid(TDir())/2;
	int nl4 = Grid(TDir())/4;	
	int l4 = 0;
	for(int l2 = 0; l2 < nl2; l2+=2){
		complexd pl = 1.;
		for(int layer = l2; layer < l2+2; ++layer){
			int newid = id + size * layer;
			pl *= dev_l2[newid] * l2norm;
		}
		int pos = id + size * l4;
		dev_l4[pos] = pl + dev_l4[pos];
		l4++;	
	}
}


class L2AvgL4ML: Tunable{
private:
	complexd *l4;
	complexd *l2;
	complexd *temp_l4;
	double l2norm;
	size_t sl4;
	int radius;
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
	kernel_l2avg_l4_multilevel<<<tp.grid, tp.block, 0, stream>>>(l2, l4, radius, l2norm);
}
public:	
   L2AvgL4ML(complexd *l2, complexd *l4, size_t sl4, int radius, double l2norm) : l2(l2), l4(l4), sl4(sl4), radius(radius), l2norm(l2norm) {
	size = SpatialVolume() * radius * (Dirs()-1);
	timesec = 0.0;  
}
   ~L2AvgL4ML(){ };
   void Run(const cudaStream_t &stream){
#ifdef TIMMINGS
    time.start();
#endif
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
	temp_l4 = (complexd*)dev_malloc(sl4*sizeof(complexd));	
	cudaSafeCall(cudaMemcpy(temp_l4, l4, sl4*sizeof(complexd), cudaMemcpyDeviceToDevice));	
  }
  void postTune() {  
	cudaSafeCall(cudaMemcpy(l4, temp_l4, sl4*sizeof(complexd), cudaMemcpyDeviceToDevice));
	dev_free(temp_l4);
 }

};



__global__ void kernel_l4avg_Final_multilevel(complexd *dev_l4, complexd *res, int radius, double norm){
    size_t id = threadIdx.x + blockDim.x * blockIdx.x;    				
	
	int nl4 = Grid(TDir())/4;	
	for(int r = 0; r < radius; ++r)	{
		complexd pp = 0.;
		if( id < SpatialVolume() ){
			for(int dir = 0; dir < TDir(); dir++){
				complexd pl = 1.;
				for(int l4 = 0; l4 < nl4; ++l4){
					int newid = id + SpatialVolume() * r + SpatialVolume() * radius * dir + SpatialVolume() * radius * (Dirs()-1) * l4;
					pl *= dev_l4[newid] * norm;
				}
				pp += pl;
			}
		}
		reduce_block_1d<complexd>(res + r, pp);
		__syncthreads();
	}
}

class L4AvgPP: Tunable{
private:
	complexd* l4;
	complexd *poly;
	complexd *dev_poly;
	int radius;
	double norm;
	double l4norm;
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
	cudaSafeCall(cudaMemset(dev_poly, 0, radius*sizeof(complexd)));
	kernel_l4avg_Final_multilevel<<<tp.grid, tp.block, tp.shared_bytes, stream>>>(l4, dev_poly, radius, l4norm);
}
public:
   L4AvgPP(complexd *l4, int radius, double l4norm) : l4(l4), radius(radius), l4norm(l4norm) {
	size = SpatialVolume();
	dev_poly = (complexd*)dev_malloc(radius*sizeof(complexd));
	poly = (complexd*)safe_malloc(radius*sizeof(complexd));
	norm = 1. / double(SpatialVolume()*(Dirs()-1));
	timesec = 0.0;  
}
   ~L4AvgPP(){ dev_free(dev_poly);};
   complexd* Run(const cudaStream_t &stream){
#ifdef TIMMINGS
    time.start();
#endif
	apply(stream);
	cudaSafeCall(cudaMemcpy(poly, dev_poly, radius*sizeof(complexd), cudaMemcpyDeviceToHost));
	for(int i = 0; i < radius; ++i) poly[i] *= norm;
    CUDA_SAFE_DEVICE_SYNC();
    CUT_CHECK_ERROR("Kernel execution failed");
#ifdef TIMMINGS
	CUDA_SAFE_DEVICE_SYNC( );
    time.stop();
    timesec = time.getElapsedTimeInSec();
#endif
	return poly;
}
   complexd* Run(){	return Run(0);}
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




complexd* MultiLevel(double *lat, cuRNGState *rng_state, int n4, int k4, int n2, int k2, int metrop, int ovrn){

	if( Grid(TDir())%4 != 0 ) {
		cout << "Error: Cannot Apply MultiLevel Algorithm...\n Nt is not multiple of 4...\n Exiting..." << endl;
		exit(1);
	}
	double *dev_lat = (double*)dev_malloc(Volume()*Dirs()*sizeof(double));
	cudaSafeCall(cudaMemcpy(dev_lat, lat, Volume()*Dirs()*sizeof(double), cudaMemcpyDeviceToDevice));

	int radius = Grid(0)/2;
	int nl2 = Grid(TDir())/2;
	int sl2 = nl2*(Dirs()-1)*radius*SpatialVolume();
	complexd *l2 = (complexd*)dev_malloc(sl2*sizeof(complexd));
	int nl4 = Grid(TDir())/4;
	size_t sl4 = nl4*(Dirs()-1)*radius*SpatialVolume();
	complexd *l4 = (complexd*)dev_malloc(sl4*sizeof(complexd));
	
	
	

	// metropolis and overrelaxation algorithm
	Metropolis_ML<4> mtp4(dev_lat, rng_state, metrop);
	OverRelaxation_ML<4> ovr4(dev_lat, ovrn);
	
	Metropolis_ML<2> mtp2(dev_lat, rng_state, metrop);
	OverRelaxation_ML<2> ovr2(dev_lat, ovrn);
	
	const bool multihit = true;
	Polyakov_Volume<multihit> mhitVol(dev_lat);
	complexd* dev_mhit;
	
	double l2norm = 1./double(n2);
	L2AvgL4ML l2avgl4(l2, l4, sl4, radius, l2norm);
	double l4norm = 1./double(n4);
	L4AvgPP l4avgpp(l4, radius, l4norm);

	cudaSafeCall(cudaMemset(l4, 0, sl4*sizeof(complexd)));
	for(int i = 0; i < n4; ++i){
		cout << "Iter of l4: " << i << endl;
		//Update the lattice k4 times freezing spacial links in layers with t multiple of 4
		for(int j = 0; j < k4; ++j){
			mtp4.Run();
			ovr4.Run();
		}
		cudaSafeCall(cudaMemset(l2, 0, sl2*sizeof(complexd)));
		for(int k = 0; k < n2; ++k){		
			//Update the lattice k2 times freezing spacial links in layers with t multiple of 2
			for(int l = 0; l < k2; ++l){
				mtp2.Run();
				ovr2.Run();	
			}
			//Extract temporal links and apply MultiHit
			dev_mhit = mhitVol.Run();			
			//Calculate tensor T2
			L2ML l2ml(dev_mhit, l2, sl2, radius);
			l2ml.Run();
		}
		//Average tensor T2 and Calculate tensor T4
		l2avgl4.Run();
		
		
		
		
		
		/*double l4norm1 = 1./double(i+1);
		L4AvgPP l4avgpp1(l4, radius, l4norm1);
		complexd* res = l4avgpp1.Run();		
		for(int r = 0; r < radius; ++r){
			cout << i << '\t' << r+1 << '\t' << res[r].real() << '\t' << res[r].imag() << endl;
		}
		cout << "################################" << endl;		
		host_free(res);*/
		
	
	}
	dev_free(dev_lat);
	dev_free(dev_mhit);
	dev_free(l2);
	//Average tensor T4 and Calculate P(0)*conj(P(r))	
	complexd* res = l4avgpp.Run();
	dev_free(l4);

	std::ofstream fileout;
	std::string filename = "Pot_mlevel_" + GetLatticeName() + ".dat";
	fileout.open (filename.c_str());
	if (!fileout.is_open()) {
		std::cout << "Error opening file: " << filename << std::endl;
		exit(1);
	}
	fileout.precision(12);
		
	for(int r = 0; r < radius; ++r){
		cout << r+1 << '\t' << res[r].real() << '\t' << res[r].imag() << endl;
		fileout << r+1 << '\t' << res[r].real() << '\t' << res[r].imag() << endl;
	}
	
	fileout.close();		
	//host_free(res);
	return(res);
} 




}
