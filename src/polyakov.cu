#include <iostream>
#include <math.h> 
#include <time.h> 
#include <random>
#include <vector> 
#include <fstream>
#include <omp.h>

#include <cuda.h>

#include "timer.h"
#include "cuda_error_check.h"
#include "alloc.h"
#include "reduce_block_1d.h"
#include "complex.h"


#include "parameters.h"
#include "index.h"
#include "array.h"


#include "staple.h"

#include "tune.h"
#include "lattice_functions.h"

using namespace std;


namespace U1{

void polyakov(double *lat, double *poly){
	for(int i = 0; i < 2; ++i) poly[i] = 0.;
	for(int parity = 0; parity < 2; ++parity){
		#pragma omp parallel for reduction(+:poly[:2])
		for(int id = 0; id < SpatialVolume()/2; ++id){
			int x[4];
			indexEO(id, parity, x);
			double tmp = 0.;
			for(x[TDir()] = 0; x[TDir()] < Grid(TDir()); ++x[TDir()])
				tmp += lat[ indexId(x, TDir()) ];
			poly[0] += cos(tmp);
			poly[1] += sin(tmp);
		}
	}
	double norm = 1. / double(SpatialVolume());
	for(int i = 0; i < 2; ++i) poly[i] *= norm;
}










__global__ void kernel_polyakov(double *lat, complexd *poly){
    size_t id = threadIdx.x + blockDim.x * blockIdx.x;

	complexd poly0 = 0.;
	
	if( id < SpatialVolume()/2 ){
		for(int parity = 0; parity < 2; ++parity){		
			int x[4];
			indexEO(id, parity, x);
			double tmp = 0.;
			for(x[TDir()] = 0; x[TDir()] < Grid(TDir()); ++x[TDir()])
				tmp += lat[ indexId(x, TDir()) ];
			poly0 += exp_ir(tmp);
		}
	}
	reduce_block_1d<complexd>(poly, poly0);
}





class CalcPolyakov: Tunable{
private:
	Array<double>* lat;
	complexd poly;
	complexd *dev_poly;
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
	cudaSafeCall(cudaMemset(dev_poly, 0, sizeof(complexd)));
	kernel_polyakov<<<tp.grid, tp.block, tp.shared_bytes, stream>>>(lat->getPtr(), dev_poly);
}
public:
   CalcPolyakov(Array<double>* lat) : lat(lat) {
	size = SpatialVolume()/2;
	dev_poly = (complexd*)dev_malloc(sizeof(complexd));
	norm = 1. / double(SpatialVolume());
	timesec = 0.0;  
}
   ~CalcPolyakov(){ dev_free(dev_poly);};
   complexd Run(const cudaStream_t &stream){
#ifdef TIMMINGS
    time.start();
#endif
	apply(stream);
	cudaSafeCall(cudaMemcpy(&poly, dev_poly, sizeof(complexd), cudaMemcpyDeviceToHost));
	poly *= norm;
    cudaDevSync();
    cudaCheckError("Kernel execution failed");
#ifdef TIMMINGS
	cudaDevSync( );
    time.stop();
    timesec = time.getElapsedTimeInSec();
#endif
	return poly;
}
   complexd Run(){	return Run(0);}
   double flops(){	return ((double)flop() * 1.0e-9) / timesec;}
   double bandwidth(){	return (double)bytes() / (timesec * (double)(1 << 30));}
   long long flop() const { return 0;}
   long long bytes() const{ return 0;}
   double get_time(){	return timesec;}
   void stat(){	cout << "CalcPolyakov:  " <<  get_time() << " s\t"  << bandwidth() << " GB/s\t" << flops() << " GFlops"  << endl;}
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



complexd Polyakov(Array<double> *dev_lat, bool print){
	CalcPolyakov pl(dev_lat);
	complexd poly = pl.Run();
	if(print) cout << "L: " << poly.real() << '\t' << poly.imag() << "\t|L|: " << poly.abs() << endl;
	return poly;
} 










template<class Real,  bool multihit>
__global__ void kernel_polyakov_volume(Real *lat, complexd *poly){
    size_t id = threadIdx.x + blockDim.x * blockIdx.x;
   
	if( id >= SpatialVolume() ) return;
	int parity = 0;
	if( id >= SpatialVolume()/2 ){
		parity = 1;	
		id -= SpatialVolume()/2;
	}	
	int x[4];
	indexEO(id, parity, x);
	
	complexd res = 1.;
	for(x[TDir()] = 0; x[TDir()] < Grid(TDir()); ++x[TDir()]){	
		if(multihit){		
			int pos = indexId(x) >> 1;
			int oddbit = GetParity(x);
			res *= MultiHit(lat, pos, oddbit, TDir());
		}
		else{
			res *= GetValue<Real>(lat[ indexId(x, TDir()) ]);
		}
	}
	poly[indexIdS(x)] = res;
}

template<class Real, bool multihit>
class Polyakov_Vol: Tunable{
private:
	Array<Real>* lat;
	Array<complexd>* poly;
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
	kernel_polyakov_volume<Real, multihit><<<tp.grid, tp.block, 0, stream>>>(lat->getPtr(), poly->getPtr());
}
public:
   Polyakov_Vol(Array<Real>* lat) : lat(lat) {
	size = SpatialVolume();
	poly = new Array<complexd>(Device, SpatialVolume() );
	timesec = 0.0;  
}
   ~Polyakov_Vol(){ };
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
	return poly;
}
   Array<complexd>* Run(){	return Run(0);}
   double flops(){	return ((double)flop() * 1.0e-9) / timesec;}
   double bandwidth(){	return (double)bytes() / (timesec * (double)(1 << 30));}
   long long flop() const { return 0;}
   long long bytes() const{ return 0;}
   double get_time(){	return timesec;}
   void stat(){	cout << "Polyakov_Vol:  " <<  get_time() << " s\t"  << bandwidth() << " GB/s\t" << flops() << " GFlops"  << endl;}
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




template<bool savePPspace>
__global__ void kernel_PP(complexd *poly, complexd *pp, complexd *ppspace, int Rmax){
    size_t id = threadIdx.x + blockDim.x * blockIdx.x;
    
	complexd pl0 = 0.;
	if(id < SpatialVolume()) pl0 = poly[id];			
	int x[3];
	indexNOSD(id, x);
	for(int r = 0; r < Rmax; ++r){	
		complexd pl = 0.;
		if(id < SpatialVolume()){
			for(int dir = 0; dir < TDir(); dir++){
				int xold = x[dir];
				x[dir] = (x[dir] + r) % Grid(dir);
				complexd pl1 = poly[indexIdS(x)];
				complexd pldir = pl0 * conj(pl1);				
				x[dir] = xold;
				if(savePPspace) ppspace[id + SpatialVolume() * dir + SpatialVolume() * (Dirs()-1) *r] = pldir;
				pl += pldir;
			}
			
		}				
		reduce_block_1d<complexd>(pp + r, pl);
		__syncthreads();
	}
}




template<bool savePPspace>
class PP: Tunable{
private:
	Array<complexd> *pvol;
	Array<complexd> *poly;
	Array<complexd> *ppspace;
	Array<complexd> *dev_poly;
	int Rmax;
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
	dev_poly->Clear();
	if(savePPspace) kernel_PP<savePPspace><<<tp.grid, tp.block, tp.shared_bytes, stream>>>(pvol->getPtr(), dev_poly->getPtr(), ppspace->getPtr(), Rmax);
	else kernel_PP<savePPspace><<<tp.grid, tp.block, tp.shared_bytes, stream>>>(pvol->getPtr(), dev_poly->getPtr(), 0, Rmax);
}
public:
   PP(Array<complexd> *pvol, int Rmax) : pvol(pvol), Rmax(Rmax) {
	size = SpatialVolume();
	dev_poly = new Array<complexd>(Device, Rmax);
	poly = new Array<complexd>(Host, Rmax);
	norm = 1. / double(SpatialVolume()*(Dirs()-1));
	if(savePPspace) ppspace = new Array<complexd>(Device, SpatialVolume() * Rmax * (Dirs()-1));
	timesec = 0.0;  
}
   ~PP(){ delete dev_poly;};
   Array<complexd>* Get_PPspace(){ return ppspace; }
   Array<complexd>* Run(const cudaStream_t &stream){
#ifdef TIMMINGS
    time.start();
#endif
	apply(stream);
	poly->Copy(dev_poly);
	for(int i = 0; i < Rmax; ++i) poly->getPtr()[i] *= norm;
    cudaDevSync();
    cudaCheckError("Kernel execution failed");
#ifdef TIMMINGS
	cudaDevSync( );
    time.stop();
    timesec = time.getElapsedTimeInSec();
#endif
	return poly;
}
   Array<complexd>* Run(){	return Run(0);}
   double flops(){	return ((double)flop() * 1.0e-9) / timesec;}
   double bandwidth(){	return (double)bytes() / (timesec * (double)(1 << 30));}
   long long flop() const { return 0;}
   long long bytes() const{ return 0;}
   double get_time(){	return timesec;}
   void stat(){	cout << "PP:  " <<  get_time() << " s\t"  << bandwidth() << " GB/s\t" << flops() << " GFlops"  << endl;}
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
Array<complexd>* Poly2(Array<Real> *lat, bool multihit){
	int Rmax = Grid(0)/2+1;
	
	Array<complexd>* poly = 0;
	if(multihit){
		Polyakov_Vol<Real, true> pvol(lat);
		poly = pvol.Run();
	}
	else{
		Polyakov_Vol<Real, false> pvol(lat);
		poly = pvol.Run();
	}
	PP<false> pp(poly, Rmax);
	Array<complexd>* poly2 = pp.Run();
	if(poly) delete poly;
	
	std::ofstream fileout;
	std::string filename = "";
	if(multihit) filename = "Pot_mhit_" + GetLatticeNameI() + ".dat";
	else filename = "Pot_" + GetLatticeNameI() + ".dat";
	fileout.open (filename.c_str());
	if (!fileout.is_open()) {
		std::cout << "Error opening file: " << filename << std::endl;
		exit(1);
	}
	fileout << std::scientific;
	fileout.precision(12);
		
	for(int r = 0; r < Rmax; ++r){
		cout << r << '\t' << poly2->at(r).real() << '\t' << poly2->at(r).imag() << endl;
		fileout << r << '\t' << poly2->at(r).real() << '\t' << poly2->at(r).imag() << endl;
	}
	
	fileout.close();	
	return poly2;
}
template Array<complexd>* Poly2<double>(Array<double> *lat, bool multihit);
template Array<complexd>* Poly2<complexd>(Array<complexd> *lat, bool multihit);




template<class Real>
void Poly2(Array<Real> *lat, Array<complexd> **poly2, Array<complexd> **ppspace, bool multihit){
	int Rmax = Grid(0)/2;
	
	Array<complexd>* poly = 0;
	if(multihit){
		Polyakov_Vol<Real, true> pvol(lat);
		poly = pvol.Run();
	}
	else{
		Polyakov_Vol<Real, false> pvol(lat);
		poly = pvol.Run();
	}
	PP<true> calc_pp(poly, Rmax);
	*poly2 = calc_pp.Run();
	*ppspace = calc_pp.Get_PPspace();
	if(poly) delete poly;
	
	std::ofstream fileout;
	std::string filename = "";
	if(multihit) filename = "Pot_mhit_" + GetLatticeNameI() + ".dat";
	else filename = "Pot_" + GetLatticeNameI() + ".dat";
	fileout.open (filename.c_str());
	if (!fileout.is_open()) {
		std::cout << "Error opening file: " << filename << std::endl;
		exit(1);
	}
	fileout << std::scientific;
	fileout.precision(12);
		
	for(int r = 0; r < Rmax; ++r){
		cout << r << '\t' << (*poly2)->at(r).real() << '\t' << (*poly2)->at(r).imag() << endl;
		fileout << r << '\t' << (*poly2)->at(r).real() << '\t' << (*poly2)->at(r).imag() << endl;
	}
	
	fileout.close();	
}
template void Poly2<double>(Array<double> *lat, Array<complexd> **poly2, Array<complexd> **ppspace, bool multihit);
template void Poly2<complexd>(Array<complexd> *lat, Array<complexd> **poly2, Array<complexd> **ppspace, bool multihit);




}

