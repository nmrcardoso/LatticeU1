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
			poly0.real() += cos(tmp);
			poly0.imag() += sin(tmp);
		}
	}
	reduce_block_1d<complexd>(poly, poly0);
}



complexd dev_polyakov(double *dev_lat, complexd *dev_poly, int threads, int blocks){
	complexd poly;
	cudaSafeCall(cudaMemset(dev_poly, 0, sizeof(complexd)));
	kernel_polyakov<<<blocks, threads, threads*sizeof(complexd)>>>(dev_lat, dev_poly);
	cudaSafeCall(cudaMemcpy(&poly, dev_poly, sizeof(complexd), cudaMemcpyDeviceToHost));
	poly /= double(SpatialVolume());
	cout << "\t\t" << "L: " << poly.real() << '\t' << poly.imag() << "\t|L|: " << poly.abs() << endl;
	return poly;
} 


using namespace U1;

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



complexd Polyakov(Array<double> *dev_lat, bool print){
	CalcPolyakov pl(dev_lat);
	complexd poly = pl.Run();
	if(print) cout << "L: " << poly.real() << '\t' << poly.imag() << "\t|L|: " << poly.abs() << endl;
	return poly;
} 






__global__ void kernel_polyakov_volume(double *lat, double *poly){
    size_t id = threadIdx.x + blockDim.x * blockIdx.x;
	if( id >= SpatialVolume() ) return;
	int parity = 0;
	if( id >= SpatialVolume()/2 ){
		parity = 1;	
		id -= SpatialVolume()/2;
	}	
	int x[4];
	indexEO(id, parity, x);
	
	double tmp = 0.;
	for(x[TDir()] = 0; x[TDir()] < Grid(TDir()); ++x[TDir()]){
		tmp += lat[ indexId(x, TDir()) ];
	}
	poly[indexIdS(x)] = tmp;

}


__global__ void kernel_poly2(double *poly, complexd *poly2, int radius){
    size_t id = threadIdx.x + blockDim.x * blockIdx.x;
    
	double pl0 = 0.;
	if(id < SpatialVolume()) pl0 = poly[id];			
	int x[3];
	indexNOSD(id, x);
	for(int r = 1; r <= radius; ++r){	
		complexd pl = 0.;
		if(id < SpatialVolume()){
			for(int dir = 0; dir < TDir(); dir++){
				int xold = x[dir];
				x[dir] = (x[dir] + r) % Grid(dir);
				double pl1 = poly[indexIdS(x)];
				pl1 = pl0-pl1;
				pl.real() += cos(pl1);
				pl.imag() += sin(pl1);	
				//pl += exp_ir(pl0) * conj(exp_ir(pl1));				
				x[dir] = xold;
			}
		}				
		reduce_block_1d<complexd>(poly2 + r - 1, pl);
		__syncthreads();
	}
}

__global__ void kernel_poly21(double *poly, complexd *poly2, int radius){
    size_t id = threadIdx.x + blockDim.x * blockIdx.x;
    
	complexd pl0 = 0.;
	if(id < SpatialVolume()) pl0 = exp_ir(poly[id]);			
	int x[3];
	indexNOSD(id, x);
	for(int r = 1; r <= radius; ++r){	
		complexd pl = 0.;
		if(id < SpatialVolume()){
			for(int dir = 0; dir < TDir(); dir++){
				int xold = x[dir];
				x[dir] = (x[dir] + r) % Grid(dir);
				double pl1 = poly[indexIdS(x)];
				//pl1 = pl0-pl1;
				//pl.real() += cos(pl1);
				//pl.imag() += sin(pl1);	
				pl += pl0*conj(exp_ir(pl1));				
				x[dir] = xold;
			}
		}				
		reduce_block_1d<complexd>(poly2 + r - 1, pl);
		__syncthreads();
	}
}


complexd* poly2(double *dev_lat){
	int radius = Grid(0)/2;
	double *dev_poly_vol = (double*)dev_malloc(SpatialVolume()*sizeof(double));
	complexd *dev_poly2 = (complexd*)dev_malloc(radius*sizeof(complexd));
	complexd *poly2 = (complexd*)safe_malloc(radius*sizeof(complexd));
	
	
	int threads = 128;
	//int blocks0 = (HalfVolume() + threads - 1) / threads;
	int blocks0 = (SpatialVolume() + threads - 1) / threads;
	int blocks1 = (SpatialVolume() + threads - 1) / threads;
	size_t smem = threads * sizeof(complexd);
	
	
	kernel_polyakov_volume<<<blocks0, threads>>>(dev_lat, dev_poly_vol);
	//cudaSafeCall(cudaMemset(dev_poly2, 0, radius*sizeof(complexd)));
	kernel_poly2<<<blocks1, threads, smem>>>(dev_poly_vol, dev_poly2, radius);
	cudaSafeCall(cudaMemcpy(poly2, dev_poly2, radius*sizeof(complexd), cudaMemcpyDeviceToHost));
	
	
	

	std::ofstream fileout;
	std::string filename = "Pot_" + GetLatticeName() + ".dat";
	fileout.open (filename.c_str());
	if (!fileout.is_open()) {
		std::cout << "Error opening file: " << filename << std::endl;
		exit(1);
	}
	fileout.precision(12);
		
	for(int r = 0; r < radius; ++r){
		poly2[r] /= double(SpatialVolume()*(Dirs()-1));
		cout << r+1 << '\t' << poly2[r].real() << '\t' << poly2[r].imag() << endl;
		fileout << r+1 << '\t' << poly2[r].real() << '\t' << poly2[r].imag() << endl;
	}
	fileout.close();	
	
	dev_free(dev_poly2);
	dev_free(dev_poly_vol);
	//host_free(poly2);
	return poly2;
} 









__global__ void kernel_polyakov_volume_mhit(double *lat, complexd *poly){
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
		int pos = indexId(x) >> 1;
		int oddbit = GetParity(x);
		res *= MultiHit(lat, pos, oddbit, TDir());
	}
	poly[indexIdS(x)] = res;

}

__global__ void kernel_poly2_mhit(complexd *poly, complexd *poly2, int radius){
    size_t id = threadIdx.x + blockDim.x * blockIdx.x;
    
	complexd pl0 = 0.;
	if(id < SpatialVolume()) pl0 = poly[id];			
	int x[3];
	indexNOSD(id, x);
	for(int r = 1; r <= radius; ++r){	
		complexd pl = 0.;
		if(id < SpatialVolume()){
			for(int dir = 0; dir < TDir(); dir++){
				int xold = x[dir];
				x[dir] = (x[dir] + r) % Grid(dir);
				complexd pl1 = poly[indexIdS(x)];
				pl += pl0 * conj(pl1);				
				x[dir] = xold;
			}
		}				
		reduce_block_1d<complexd>(poly2 + r - 1, pl);
		__syncthreads();
	}
}


complexd* poly2_mhit(double *dev_lat){
	int radius = Grid(0)/2;
	complexd *dev_poly_vol = (complexd*)dev_malloc(SpatialVolume()*sizeof(complexd));
	complexd *dev_poly2 = (complexd*)dev_malloc(radius*sizeof(complexd));
	complexd *poly2 = (complexd*)safe_malloc(radius*sizeof(complexd));
		
	int threads = 128;
	//int blocks0 = (HalfVolume() + threads - 1) / threads;
	int blocks0 = (SpatialVolume() + threads - 1) / threads;
	int blocks1 = (SpatialVolume() + threads - 1) / threads;
	size_t smem = threads * sizeof(complexd);
	
	
	kernel_polyakov_volume_mhit<<<blocks0, threads>>>(dev_lat, dev_poly_vol);
	//cudaSafeCall(cudaMemset(dev_poly2, 0, radius*sizeof(complexd)));
	kernel_poly2_mhit<<<blocks1, threads, smem>>>(dev_poly_vol, dev_poly2, radius);
	cudaSafeCall(cudaMemcpy(poly2, dev_poly2, radius*sizeof(complexd), cudaMemcpyDeviceToHost));
	
	std::ofstream fileout;
	std::string filename = "Pot_mhit_" + GetLatticeName() + ".dat";
	fileout.open (filename.c_str());
	if (!fileout.is_open()) {
		std::cout << "Error opening file: " << filename << std::endl;
		exit(1);
	}
	fileout.precision(12);
		
	for(int r = 0; r < radius; ++r){
		poly2[r] /= double(SpatialVolume()*(Dirs()-1));
		cout << r+1 << '\t' << poly2[r].real() << '\t' << poly2[r].imag() << endl;
		fileout << r+1 << '\t' << poly2[r].real() << '\t' << poly2[r].imag() << endl;
	}
	
	fileout.close();	
	
	
	
	
	//host_free(poly2);
	dev_free(dev_poly2);
	dev_free(dev_poly_vol);
	return poly2;
} 




















template< bool multihit>
__global__ void kernel_polyakov_volume(double *lat, complexd *poly){
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
			res *= exp_ir(lat[ indexId(x, TDir()) ]);
		}
	}
	poly[indexIdS(x)] = res;
}

template< bool multihit>
class Polyakov_Vol: Tunable{
private:
	Array<double>* lat;
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
	kernel_polyakov_volume<multihit><<<tp.grid, tp.block, tp.shared_bytes, stream>>>(lat->getPtr(), poly->getPtr());
}
public:
   Polyakov_Vol(Array<double>* lat) : lat(lat) {
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





__global__ void kernel_PP(complexd *poly, complexd *res, int radius){
    size_t id = threadIdx.x + blockDim.x * blockIdx.x;
    
	complexd pl0 = 0.;
	if(id < SpatialVolume()) pl0 = poly[id];			
	int x[3];
	indexNOSD(id, x);
	for(int r = 1; r <= radius; ++r){	
		complexd pl = 0.;
		if(id < SpatialVolume()){
			for(int dir = 0; dir < TDir(); dir++){
				int xold = x[dir];
				x[dir] = (x[dir] + r) % Grid(dir);
				complexd pl1 = poly[indexIdS(x)];
				pl += pl0 * conj(pl1);				
				x[dir] = xold;
			}
		}				
		reduce_block_1d<complexd>(res + r - 1, pl);
		__syncthreads();
	}
}




class PP: Tunable{
private:
	Array<complexd> *pvol;
	Array<complexd> *poly;
	Array<complexd> *dev_poly;
	int radius;
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
	kernel_PP<<<tp.grid, tp.block, tp.shared_bytes, stream>>>(pvol->getPtr(), dev_poly->getPtr(), radius);
}
public:
   PP(Array<complexd> *pvol, int radius) : pvol(pvol), radius(radius) {
	size = SpatialVolume();
	dev_poly = new Array<complexd>(Device, radius);
	poly = new Array<complexd>(Host, radius);
	norm = 1. / double(SpatialVolume()*(Dirs()-1));
	timesec = 0.0;  
}
   ~PP(){ delete dev_poly;};
   Array<complexd>* Run(const cudaStream_t &stream){
#ifdef TIMMINGS
    time.start();
#endif
	apply(stream);
	poly->Copy(dev_poly);
	for(int i = 0; i < radius; ++i) poly->getPtr()[i] *= norm;
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




Array<complexd>* Poly2(Array<double> *lat, bool multihit){
	int radius = Grid(0)/2;
	
	Array<complexd>* poly = 0;
	if(multihit){
		Polyakov_Vol<true> pvol(lat);
		poly = pvol.Run();
	}
	else{
		Polyakov_Vol<false> pvol(lat);
		poly = pvol.Run();
	}
	PP pp(poly, radius);
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
	fileout.precision(12);
		
	for(int r = 0; r < radius; ++r){
		cout << r+1 << '\t' << poly2->getPtr()[r].real() << '\t' << poly2->getPtr()[r].imag() << endl;
		fileout << r+1 << '\t' << poly2->getPtr()[r].real() << '\t' << poly2->getPtr()[r].imag() << endl;
	}
	
	fileout.close();	
	
	//host_free(poly2);
	return poly2;
} 


}

