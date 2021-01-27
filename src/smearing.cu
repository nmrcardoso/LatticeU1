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
#include "lattice_functions.h"
#include "array.h"


using namespace std;


namespace U1{





template<class Real, int option>
__global__ void kernel_smearing_MultiHit(Real *lat, complexd *out){
    size_t id = threadIdx.x + blockDim.x * blockIdx.x;
   
	if( id >= Volume() ) return;
	
	int parity = 0;
	if( id >= HalfVolume() ){
		parity = 1;	
		id -= HalfVolume();
	}	
	// Apply multihit only in space links
	if(option == 0){
		for(int mu = 0; mu < Dirs()-1; mu++)
			out[id + parity * HalfVolume() + mu * Volume()] = MultiHit(lat, id, parity, mu);
		out[id + parity * HalfVolume() + TDir() * Volume()] = GetValue(lat[id + parity * HalfVolume() + TDir() * Volume()]);
	}
	// Apply multihit only in time links
	if(option == 1){
		for(int mu = 0; mu < Dirs()-1; mu++) 
			out[id + parity * HalfVolume() + mu * Volume()] = GetValue(lat[id + parity * HalfVolume() + mu * Volume()]);
		out[id + parity * HalfVolume() + TDir() * Volume()] = MultiHit(lat, id, parity, TDir());
	}
	// Apply multihit all links
	if(option == 2) {
		for(int mu = 0; mu < Dirs(); mu++)
			out[id + parity * HalfVolume() + mu * Volume()] = MultiHit(lat, id, parity, mu);
	}
	if(option == 3) {
		for(int mu = 0; mu < Dirs(); mu++) 
			out[id + parity * HalfVolume() + mu * Volume()] = GetValue(lat[id + parity * HalfVolume() + mu * Volume()]);
	}
}

template<class Real, int option>
class MHit: Tunable{
private:
	Array<Real>* lat;
	Array<complexd>* out;
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
	kernel_smearing_MultiHit<Real, option><<<tp.grid, tp.block, 0, stream>>>(lat->getPtr(), out->getPtr());
}
public:
   MHit(Array<Real>* lat, Array<complexd>* out) : lat(lat), out(out) {
	size = Volume();
	timesec = 0.0;  
}
   ~MHit(){ };
   void Run(const cudaStream_t &stream){
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
}
   void Run(){	Run(0);}
   double flops(){	return ((double)flop() * 1.0e-9) / timesec;}
   double bandwidth(){	return (double)bytes() / (timesec * (double)(1 << 30));}
   long long flop() const { return 0;}
   long long bytes() const{ return 0;}
   double time(){	return timesec;}
   void stat(){	cout << "MultiHit:  " <<  time() << " s\t"  << bandwidth() << " GB/s\t" << flops() << " GFlops"  << endl;}
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


//option 0: Apply multihit only in space links
//option 1: Apply multihit only in time links
//option 2: Apply multihit all links
//option other: convert phase(double) to gauge link (complex double) or just copy if complex double
template<class Real>
Array<complexd>* ApplyMultiHit(Array<Real>* lat, int option){
	Array<complexd>* out = new Array<complexd>(Device, lat->Size());
	switch(option){
		case 0:
		{std::cout << "Apply MultiHit only in space links..." << std::endl;
		MHit<Real, 0> mhit(lat, out);
		mhit.Run();}
		break;
		case 1:
		{std::cout << "Apply MultiHit only in time links..." << std::endl;
		MHit<Real, 1> mhit(lat, out);
		mhit.Run();}
		break;
		case 2:
		{std::cout << "Apply MultiHit all links..." << std::endl;
		MHit<Real, 2> mhit(lat, out);
		mhit.Run();}
		break;
		default:
		{std::cout << "Just copy the lattice..." << std::endl;
		MHit<Real, 3> mhit(lat, out);
		mhit.Run();}
		break;
	}
	return out;
}
template Array<complexd>* ApplyMultiHit<double>(Array<double>* lat, int option);
template Array<complexd>* ApplyMultiHit<complexd>(Array<complexd>* lat, int option);






















InlineHostDevice complexd smear_ape(complexd *lat, const int id, const int parity, const int mu, double w){
	complexd staple = Staple(lat, id, parity, mu);
	int pos = id + parity * HalfVolume() + mu * Volume();
	return exp_ir((GetValue(lat[pos]) + conj(staple) * w).phase()); //<----- NEED TO CHECK THIS
}
InlineHostDevice double smear_ape(double *lat, const int id, const int parity, const int mu, double w){
	complexd staple = Staple(lat, id, parity, mu);
	int pos = id + parity * HalfVolume() + mu * Volume();
	return (GetValue(lat[pos]) + conj(staple) * w).phase(); //<----- NEED TO CHECK THIS
}




template<class Real, int option>
__global__ void kernel_smearing_APE(Real *lat, Real *out, double w){
    size_t id = threadIdx.x + blockDim.x * blockIdx.x;
   
	if( id >= Volume() ) return;
	
	int parity = 0;
	if( id >= HalfVolume() ){
		parity = 1;	
		id -= HalfVolume();
	}	
	// Apply multihit only in space links
	if(option == 0){
		for(int mu = 0; mu < Dirs()-1; mu++)
			out[id + parity * HalfVolume() + mu * Volume()] = smear_ape(lat, id, parity, mu, w);
	}
	// Apply multihit only in time links
	if(option == 1){
		out[id + parity * HalfVolume() + TDir() * Volume()] = smear_ape(lat, id, parity, TDir(), w);
	}
	// Apply multihit all links
	if(option == 2) {
		for(int mu = 0; mu < Dirs(); mu++)
			out[id + parity * HalfVolume() + mu * Volume()] = smear_ape(lat, id, parity, mu, w);
	}
}


template<class Real, bool option>
class APE: Tunable{
private:
	Array<Real>* lat;
	Array<Real>* out;
	Array<Real>* in;
	double w;
	int n;
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
	kernel_smearing_APE<Real, option><<<tp.grid, tp.block, 0, stream>>>(in->getPtr(), out->getPtr(), w);
}
public:
   APE(Array<Real>* lat, Array<Real>* out, double w, int n=1) : lat(lat), out(out), w(w), n(n) {
	size = Volume();
	in = new Array<Real>(Device, lat->Size());
	in->Copy(lat);
	timesec = 0.0;  
}
   ~APE(){ delete in; };
   void Run(const cudaStream_t &stream){
#ifdef TIMMINGS
    time.start();
#endif
	for(int i = 0; i < n; i++){
		apply(stream);
		in->Copy(out);	
	}
	apply(stream);
    cudaDevSync();
    cudaCheckError("Kernel execution failed");
#ifdef TIMMINGS
	cudaDevSync( );
    time.stop();
    timesec = time.getElapsedTimeInSec();
#endif
}
   void Run(){	Run(0);}
   double flops(){	return ((double)flop() * 1.0e-9) / timesec;}
   double bandwidth(){	return (double)bytes() / (timesec * (double)(1 << 30));}
   long long flop() const { return 0;}
   long long bytes() const{ return 0;}
   double time(){	return timesec;}
   void stat(){	cout << "MultiHit:  " <<  time() << " s\t"  << bandwidth() << " GB/s\t" << flops() << " GFlops"  << endl;}
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




//option 0: Apply multihit only in space links
//option 1: Apply multihit only in time links
//option 2: Apply multihit all links
template<class Real>
Array<Real>* ApplyAPE(Array<Real>* lat, double w, int n, int option){
	Array<Real>* out = new Array<Real>(Device, lat->Size());
	out->Copy(lat);
	switch(option){
		case 0:
		{std::cout << "Apply APE only in space links..." << std::endl;
		std::cout << "Factor: " << w << "\tNiter: " << n << std::endl;
		APE<Real, 0> ape(lat, out, w, n);
		ape.Run();}
		break;
		case 1:
		{std::cout << "Apply APE only in time links..." << std::endl;
		std::cout << "Factor: " << w << "\tNiter: " << n << std::endl;
		APE<Real, 1> ape(lat, out, w, n);
		ape.Run();}
		break;
		case 2:
		{std::cout << "Apply APE all links..." << std::endl;
		std::cout << "Factor: " << w << "\tNiter: " << n << std::endl;
		APE<Real, 2> ape(lat, out, w, n);
		ape.Run();}
		break;
		default:
		{std::cout << "Option not valid..." << std::endl;
		Finalize(1);}
		break;
	}
	return out;
}

template Array<double>* ApplyAPE<double>(Array<double>* lat, double w, int n, int option);
template Array<complexd>* ApplyAPE<complexd>(Array<complexd>* lat, double w, int n, int option);








 

}
