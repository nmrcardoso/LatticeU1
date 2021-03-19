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
__global__ void kernel_WilsonSPLines(Real *lat, complexd *wlinesp, int Rmax){
    size_t id = threadIdx.x + blockDim.x * blockIdx.x;
	if(id < Volume()){
		for(int dir = 0; dir < Dirs()-1; dir++){
			complexd wline = 1.0;			
			int idx = id;
			wlinesp[id + Volume() * dir] = wline;
			for(int r = 0; r < Rmax-1; r++){	
				wline *= GetValue(lat[idx + dir * Volume()]);
				idx = indexNO_neg(idx, dir, 1);
				wlinesp[id + Volume() * dir + Volume() * (Dirs()-1) * (r+1)] = wline; 
			}
		}
	}   
}


template< class Real>
class WilsonSPLines: Tunable{
private:
	Array<Real>* lat;
	Array<complexd>* wlinesp;
	int R;
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
	wlinesp->Clear();
	kernel_WilsonSPLines<Real><<<tp.grid, tp.block, tp.shared_bytes, stream>>>(lat->getPtr(), wlinesp->getPtr(), R);
}
public:
   WilsonSPLines(Array<Real>* lat, int R) : lat(lat), R(R) {
	size = Volume();
	wlinesp = new Array<complexd>(Device, Volume() * (Dirs()-1) * R);
	timesec = 0.0;  
}
   ~WilsonSPLines(){ delete wlinesp; };
   Array<complexd>* Run(const cudaStream_t &stream){
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
	return wlinesp;
}
   Array<complexd>* Run(){ return Run(0);}
   double flops(){	return ((double)flop() * 1.0e-9) / timesec;}
   double bandwidth(){	return (double)bytes() / (timesec * (double)(1 << 30));}
   long long flop() const { return 0;}
   long long bytes() const{ return 0;}
   double get_time(){	return timesec;}
   void stat(){	cout << "WilsonSPLines:  " <<  get_time() << " s\t"  << bandwidth() << " GB/s\t" << flops() << " GFlops"  << endl;}
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



template< class Real, bool getfield>
__global__ void kernel_wilsonloopPreSPWL(Real *lat, complexd *wlinesp, complexd *wloop, complexd *wloopField, int Rmax, int Tmax){
    size_t id = threadIdx.x + blockDim.x * blockIdx.x;
	for(int dir = 0; dir < Dirs()-1; dir++)
	for(int r = 0; r < Rmax; r++){
		complexd left = 1.0;
		complexd right = 1.0;
		complexd top = 1.0;
		complexd bottom = 1.0;
		
		if(id < Volume()) bottom = wlinesp[id + Volume() * dir + Volume() * (Dirs()-1) * r];
		
		int idl = id;
		int idr = indexNO_neg(id, dir, r);
	
		//Começa aqui a confusão
		for( int t = 0; t < Tmax; ++t ) {

			complexd top = 1.0;
			int idt = indexNO_neg(id, TDir(), t);			
			if(id < Volume()) top = wlinesp[idt + Volume() * dir + Volume() * (Dirs()-1) * r];
    
    		complexd wl = 0.0;
    		if(id < Volume()){
    			wl = bottom * right * conj(top) * conj(left);
    			if(getfield) wloopField[id + Volume() * dir + Volume() * (Dirs()-1) * (r + Rmax * t)] = wl;
			}
    		
			reduce_block_1d<complexd>(wloop + t + Tmax * r, wl);
  		
			//actualiza linhas temporais
			if(id < Volume()){
				left *= GetValue(lat[idl + TDir() * Volume()]);
				right *= GetValue(lat[idr + TDir() * Volume()]);
			}
			idl = indexNO_neg(idl, TDir(), 1);
			idr = indexNO_neg(idr, TDir(), 1);
		}
	}   
}
    
   

template< class Real, bool getfield>
class wilsonloopWPreWL: Tunable{
private:
	Array<Real>* lat;
	Array<complexd>* wlinesp;
	Array<complexd>* wloopField;
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
	if(getfield) kernel_wilsonloopPreSPWL<Real, getfield><<<tp.grid, tp.block, tp.shared_bytes, stream>>>(lat->getPtr(), wlinesp->getPtr(), wloop_dev->getPtr(), wloopField->getPtr(), R, T);
	else kernel_wilsonloopPreSPWL<Real, getfield><<<tp.grid, tp.block, tp.shared_bytes, stream>>>(lat->getPtr(), wlinesp->getPtr(), wloop_dev->getPtr(), 0, R, T);
}
public:
   wilsonloopWPreWL(Array<Real>* lat, Array<complexd>* wlinesp, int R, int T) : lat(lat), wlinesp(wlinesp), R(R), T(T) {
	size = Volume();
	wloop = new Array<complexd>(Host, R*T);
	wloop_dev = new Array<complexd>(Device, R*T);
	if(getfield) wloopField = new Array<complexd>(Device, R*T*Volume()*(Dirs()-1));  
	timesec = 0.0;  
}
   ~wilsonloopWPreWL(){ delete wloop_dev; };
   Array<complexd>* GetField(){ return wloopField; }
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
   double get_time(){	return timesec;}
   void stat(){	cout << "wilsonloopWPreWL:  " <<  get_time() << " s\t"  << bandwidth() << " GB/s\t" << flops() << " GFlops"  << endl;}
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






		
		
template< class Real, bool getfield>
__global__ void kernel_wilsonloop(Real *lat, complexd *wloop, complexd *wloopField, int Rmax, int Tmax){
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
    		if(id < Volume()){
    			wl = bottom * right * conj(top) * conj(left);
    			if(getfield) wloopField[id + Volume() * dir + Volume() * (Dirs()-1) * (r + Rmax * t)] = wl;
			}
    		
			reduce_block_1d<complexd>(wloop + t + Tmax * r, wl);
     		
			//actualiza linhas temporais
			if(id < Volume()){
				left *= GetValue(lat[idl + TDir() * Volume()]);
				right *= GetValue(lat[idr + TDir() * Volume()]);
			}
			idl = indexNO_neg(idl, TDir(), 1);
			idr = indexNO_neg(idr, TDir(), 1);
		}
	}   
}
    
   

template< class Real, bool getfield>
class wilsonloop: Tunable{
private:
	Array<Real>* lat;
	Array<complexd>* wloop;
	Array<complexd>* wloopField;
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
	if(getfield) kernel_wilsonloop<Real, getfield><<<tp.grid, tp.block, tp.shared_bytes, stream>>>(lat->getPtr(), wloop_dev->getPtr(), wloopField->getPtr(), R, T);
	else kernel_wilsonloop<Real, getfield><<<tp.grid, tp.block, tp.shared_bytes, stream>>>(lat->getPtr(), wloop_dev->getPtr(), 0, R, T);
}
public:
   wilsonloop(Array<Real>* lat, int R, int T) : lat(lat), R(R), T(T) {
	size = Volume();
	wloop = new Array<complexd>(Host, R*T);
	wloop_dev = new Array<complexd>(Device, R*T);
	if(getfield) wloopField = new Array<complexd>(Device, R*T*Volume()*(Dirs()-1));
	timesec = 0.0;  
}
   ~wilsonloop(){ delete wloop_dev; };
   Array<complexd>* GetField(){ return wloopField; }
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
   double get_time(){	return timesec;}
   void stat(){	cout << "Wilson loop:  " <<  get_time() << " s\t"  << bandwidth() << " GB/s\t" << flops() << " GFlops"  << endl;}
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
Array<complexd>* WilsonLoop(Array<Real>* lat, int R, int T, bool FastVersion){
	if(FastVersion){
		Array<Real>* tmp = LatticeConvert(lat, true);
		WilsonSPLines<Real> wlines(tmp, R);
		Array<complexd>* wlinesp = wlines.Run();
		wilsonloopWPreWL<Real, false> wloop(tmp, wlinesp, R, T);
		Array<complexd>* res = wloop.Run();
		delete tmp;
		return res;
	}
	else{
		Array<Real>* tmp = LatticeConvert(lat, true);
		wilsonloop<Real, false> wloop(tmp, R, T);
		Array<complexd>* res = wloop.Run();
		delete tmp;
		return res;
	}
}

template Array<complexd>* WilsonLoop(Array<double>* lat, int R, int T, bool FastVersion);
template Array<complexd>* WilsonLoop(Array<complexd>* lat, int R, int T, bool FastVersion);





template<class Real>
void WilsonLoop(Array<Real>* lat, Array<complexd>** wl, Array<complexd>** wlfield, int R, int T, bool FastVersion){
	Array<Real>* tmp = LatticeConvert(lat, true);
	if(FastVersion){
		WilsonSPLines<Real> wlines(tmp, R);
		Array<complexd>* wlinesp = wlines.Run();
		wilsonloopWPreWL<Real, true> wloop(tmp, wlinesp, R, T);
		*wl = wloop.Run();
		*wlfield = wloop.GetField();
	}
	else{
		Array<Real>* tmp = LatticeConvert(lat, true);
		wilsonloop<Real, true> wloop(tmp, R, T);
		*wl = wloop.Run();
		*wlfield = wloop.GetField();
		delete tmp;
	}
	delete tmp;
}
template void WilsonLoop(Array<double>* lat, Array<complexd>** wl, Array<complexd>** wlfield, int R, int T, bool FastVersion);
template void WilsonLoop(Array<complexd>* lat, Array<complexd>** wl, Array<complexd>** wlfield, int R, int T, bool FastVersion);








































template< class Real>
__global__ void kernel_wilsonloopPreSPWL_SS(Real *lat, complexd *wlinesp, complexd *wloop, int Rmax, int Tmax){
    size_t id = threadIdx.x + blockDim.x * blockIdx.x;
	for(int dir = 0; dir < Dirs()-1; dir++)
	for(int dir1 = 0; dir1 < Dirs()-1; dir1++){
		if(dir == dir1) continue;
		for(int r = 0; r < Rmax; r++){
			complexd left = 1.0;
			complexd right = 1.0;
			complexd top = 1.0;
			complexd bottom = 1.0;
			
			if(id < Volume()) bottom = wlinesp[id + Volume() * dir + Volume() * (Dirs()-1) * r];
			
			int idl = id;
			int idr = indexNO_neg(id, dir, r);
		
			//Começa aqui a confusão
			for( int t = 0; t < Tmax; ++t ) {

				complexd top = 1.0;
				int idt = indexNO_neg(id, dir1, t);			
				if(id < Volume()){ 
					top = wlinesp[idt + Volume() * dir + Volume() * (Dirs()-1) * r];
					left = wlinesp[idl + Volume() * dir1 + Volume() * (Dirs()-1) * t];
					right = wlinesp[idr + Volume() * dir1 + Volume() * (Dirs()-1) * t];
				}
		
				complexd wl = 0.0;
				if(id < Volume()){
					wl = bottom * right * conj(top) * conj(left);
				}
				
				reduce_block_1d<complexd>(wloop + t + Tmax * r, wl);
	  		
	  			/*
				//actualiza linhas temporais
				if(id < Volume()){
					left *= GetValue(lat[idl + dir1 * Volume()]);
					right *= GetValue(lat[idr + dir1 * Volume()]);
				}
				idl = indexNO_neg(idl, dir1, 1);
				idr = indexNO_neg(idr, dir1, 1);
				*/
			}
		}  
	} 
}
    
   

template< class Real>
class wilsonloopWPreWL_SS: Tunable{
private:
	Array<Real>* lat;
	Array<complexd>* wlinesp;
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
	kernel_wilsonloopPreSPWL_SS<Real><<<tp.grid, tp.block, tp.shared_bytes, stream>>>(lat->getPtr(), wlinesp->getPtr(), wloop_dev->getPtr(), R, T);
}
public:
   wilsonloopWPreWL_SS(Array<Real>* lat, Array<complexd>* wlinesp, int R, int T) : lat(lat), wlinesp(wlinesp), R(R), T(T) {
	size = Volume();
	wloop = new Array<complexd>(Host, R*T);
	wloop_dev = new Array<complexd>(Device, R*T);
	timesec = 0.0;  
}
   ~wilsonloopWPreWL_SS(){ delete wloop_dev; };
   Array<complexd>* Run(const cudaStream_t &stream){
#ifdef TIMMINGS
    time.start();
#endif
	apply(stream);
	wloop->Copy(wloop_dev);
	for(int i = 0; i < wloop->Size(); i++) wloop->at(i) /= double(Volume()*(Dirs()-1)*2.0);
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
   double get_time(){	return timesec;}
   void stat(){	cout << "wilsonloopWPreWL:  " <<  get_time() << " s\t"  << bandwidth() << " GB/s\t" << flops() << " GFlops"  << endl;}
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
Array<complexd>* WilsonLoopSS(Array<Real>* lat, int R, int T){
	Array<Real>* tmp = LatticeConvert(lat, true);
	WilsonSPLines<Real> wlines(tmp, R);
	Array<complexd>* wlinesp = wlines.Run();
	wilsonloopWPreWL_SS<Real> wloop(tmp, wlinesp, R, T);
	Array<complexd>* res = wloop.Run();
	delete tmp;
	return res;
}

template Array<complexd>* WilsonLoopSS(Array<double>* lat, int R, int T);
template Array<complexd>* WilsonLoopSS(Array<complexd>* lat, int R, int T);












}
