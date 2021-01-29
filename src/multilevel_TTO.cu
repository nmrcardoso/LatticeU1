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

#include "array.h"
#include "multilevel.h"
#include "lattice_functions.h"

using namespace std;

namespace U1{

namespace ML_TTO{


#include "multilevel_common.cuh"



//Defined only for charges along z direction
inline __host__ __device__ void GetFields(const complexd *plaqfield, int pos, int dirx, int diry, int dirz, bool evenradius, complexd field[6]) {	
	if(evenradius){
		//Ex
		complexd plaq = plaqfield[pos + dirx * Volume()];
		int s1 = indexNO_neg(pos, dirx, -1);
		plaq += plaqfield[s1 + dirx * Volume()];
		field[0] = plaq * 0.5;
		//Ey
		plaq = plaqfield[pos + diry * Volume()];
		s1 = indexNO_neg(pos, diry, -1);
		plaq += plaqfield[s1 + diry * Volume()];
		field[1] = plaq * 0.5;
		//Ez
		plaq = plaqfield[pos + dirz * Volume()];
		s1 = indexNO_neg(pos, dirz, -1);
		plaq += plaqfield[s1 + dirz * Volume()];
		field[2] = plaq * 0.5;
		//Bx
		plaq = plaqfield[pos + (3 + dirx) * Volume()];
		s1 = indexNO_neg(pos, dirz, -1);
		plaq += plaqfield[s1 + (3 + dirx) * Volume()];
		s1 = indexNO_neg(pos, diry, -1);
		plaq += plaqfield[s1 + (3 + dirx) * Volume()];
		s1 = indexNO_neg(s1, dirz, -1);
		plaq += plaqfield[s1 + (3 + dirx) * Volume()];
		field[3] = plaq * 0.25;
		//By
		plaq = plaqfield[pos + (3 + diry) * Volume()];
		s1 = indexNO_neg(pos, dirz, -1);
		plaq += plaqfield[s1 + (3 + diry) * Volume()];
		s1 = indexNO_neg(pos, dirx, -1);
		plaq += plaqfield[s1 + (3 + diry) * Volume()];
		s1 = indexNO_neg(s1, dirz, -1);
		plaq += plaqfield[s1 + (3 + diry) * Volume()];
		field[4] = plaq * 0.25;
		//Bz
		plaq = plaqfield[pos + (3 + dirz) * Volume()];
		s1 = indexNO_neg(pos, dirx, -1);
		plaq += plaqfield[s1 + (3 + dirz) * Volume()];
		s1 = indexNO_neg(pos, diry, -1);
		plaq += plaqfield[s1 + (3 + dirz) * Volume()];
		s1 = indexNO_neg(s1, dirx, -1);
		plaq += plaqfield[s1 + (3 + dirz) * Volume()];
		field[5] = plaq * 0.25;
	}
	else{
		//Valid for mid and charge plane
		//Valid only for odd radius
		//Ex
		complexd plaq = plaqfield[pos + dirx * Volume()];
		int s1 = indexNO_neg(pos, dirx, -1);
		plaq += plaqfield[s1 + dirx * Volume()];
		s1 = indexNO_neg(pos, dirz, 1);
		plaq += plaqfield[s1 + dirx * Volume()];
		s1 = indexNO_neg(s1, dirx, -1);
		plaq += plaqfield[s1 + dirx * Volume()];
		field[0] = plaq * 0.25;
		//Ey
		plaq = plaqfield[pos + diry * Volume()];
		s1 = indexNO_neg(pos, diry, -1);
		plaq += plaqfield[s1 + diry * Volume()];
		s1 = indexNO_neg(pos, dirz, 1);
		plaq += plaqfield[s1 + diry * Volume()];
		s1 = indexNO_neg(s1, diry, -1);
		plaq += plaqfield[s1 + diry * Volume()];
		field[1] = plaq * 0.25;
		//Ez
		plaq = plaqfield[pos + dirz * Volume()];
		field[2] = plaq;
		//Bx
		plaq = plaqfield[pos + (3 + dirx) * Volume()];
		s1 = indexNO_neg(s1, diry, -1);
		plaq += plaqfield[s1 + (3 + dirx) * Volume()];
		field[3] = plaq * 0.5;
		//By
		plaq = plaqfield[pos + (3 + diry) * Volume()];
		s1 = indexNO_neg(s1, dirx, -1);
		plaq += plaqfield[s1 + (3 + diry) * Volume()];
		field[4] = plaq * 0.5;
		//Bz
		plaq = plaqfield[pos + (3 + dirz) * Volume()]; 
		s1 = indexNO_neg(pos, dirx, -1);
		plaq += plaqfield[s1 + (3 + dirz) * Volume()]; 
		s1 = indexNO_neg(pos, diry, -1);
		plaq += plaqfield[s1 + (3 + dirz) * Volume()];
		s1 = indexNO_neg(s1, dirx, -1);
		plaq += plaqfield[s1 + (3 + dirz) * Volume()]; 
		int s2 = indexNO_neg(pos, dirz, 1);
		plaq += plaqfield[s2 + (3 + dirz) * Volume()]; 
		s1 = indexNO_neg(s2, dirx, -1);
		plaq += plaqfield[s1 + (3 + dirz) * Volume()];
		s1 = indexNO_neg(s2, diry, -1);
		plaq += plaqfield[s1 + (3 + dirz) * Volume()];
		s1 = indexNO_neg(s1, dirx, -1);
		plaq += plaqfield[s1 + (3 + dirz) * Volume()];
		field[5] = plaq * 0.125;
	}
}


__global__ void kernel_l2_multilevel_1(complexd *plaqfield, complexd *poly, complexd *l2, complexd *lo2, int radius, bool SquaredField, bool alongCharges, int perpPoint){
    size_t id = threadIdx.x + blockDim.x * blockIdx.x;    
	if(id >= SpatialVolume()) return;		
	int x[4];
	indexNOSD(id, x);
	
	int nl2 = Grid(TDir())/2;
	for(int dir = 0; dir < TDir(); dir++){		
		int layer = 0;
		for(int t = 0; t < Grid(TDir()); t+=2){
			complexd pl0 = 1.;
			complexd pl1 = 1.;
			for(x[TDir()] = t; x[TDir()] < t+2; ++x[TDir()]){
				pl0 *= (poly[(((x[3] * Grid(2) + x[2]) * Grid(1)) + x[1] ) * Grid(0) + x[0]]);
				int xold = x[dir];
				x[dir] = (x[dir] + radius) % Grid(dir);
				pl1 *= conj(poly[(((x[3] * Grid(2) + x[2]) * Grid(1)) + x[1] ) * Grid(0) + x[0]]);
								
				x[dir] = xold;
			}			
			complexd pl = pl0 * pl1;
			
			for(int tt = 0; tt < 2; ++tt)
			for(int iz = 0; iz < Grid(0); ++iz){	
				complexd plaq = 1.0;
				x[TDir()] = t + tt;				
				
				int rad = radius/2;				
				//int evenradius = !(radius%2);	
				int evenradius = !(radius & 1);
								
				int dirz = dir;				//Z
				int dirx = (dirz+1)%TDir();  //X
				int diry = (dirx+1)%TDir(); //Y		
				
				int pos = 0;
				complexd field[6];
				if(alongCharges){
					int xold = x[dirz];
					int xold1 = x[dirx];
					x[dirz] = (x[dirz] + rad + iz - Grid(dirz)/2 + Grid(dirz)) % Grid(dirz);									
					x[dirx] = (x[dirx] + perpPoint + Grid(dirx)) % Grid(dirx);
					pos = indexId(x); 	
					x[dirz] = xold;
					x[dirx] = xold1;
				}
				else{
					int xold = x[dirz];
					int xold1 = x[dirx];
					int xold2 = x[diry];
					x[dirz] = (x[dirz] + rad) % Grid(dirz);					
					x[dirx] = (x[dirx] + iz - Grid(dirx)/2 + Grid(dirx)) % Grid(dirx);
					x[diry] = (x[diry] + perpPoint + Grid(diry)) % Grid(diry);
					pos = indexId(x); 	
					x[dirz] = xold;
					x[dirx] = xold1;
					x[diry] = xold2;
				}	
				GetFields(plaqfield, pos, dirx, diry, dirz, evenradius, field); 
				
				if(SquaredField){
					for(int fi = 0; fi < 6; fi++)
						field[fi].imag() = 0.0;
				}
				else {
					for(int fi = 0; fi < 6; fi++)
						field[fi].real() = 0.0;
				}
				int idout = id + SpatialVolume() * dir + SpatialVolume() * (Dirs()-1) * layer;
				idout +=  SpatialVolume() * (Dirs()-1) * nl2 * tt + SpatialVolume() * (Dirs()-1) * nl2 * 2 * iz;
				for(int fi = 0; fi < 6; fi++){
					field[fi] = pl * field[fi];
					int idout1 = idout + SpatialVolume() * (Dirs()-1) * nl2 * 2 * Grid(0) * fi;
					lo2[idout1] += field[fi];
				}
			}					
			int idout = id + SpatialVolume() * dir + SpatialVolume() * (Dirs()-1) * layer;
			l2[idout] = pl + l2[idout];
			layer++;
		}
	}
}

























class L2ML: Tunable{
private:
	Array<complexd> *plaqfield;
	Array<complexd> *poly;
	Array<complexd> *l2;
	Array<complexd> *lo2;
	size_t sl2;
	int radius;
	int size;
	int perpPoint;
	bool SquaredField;
	bool alongCharges;
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
	kernel_l2_multilevel_1<<<tp.grid, tp.block, tp.shared_bytes, stream>>>(plaqfield->getPtr(), poly->getPtr(), l2->getPtr(), lo2->getPtr(), radius, SquaredField, alongCharges, perpPoint);
}
public:	
   L2ML(Array<complexd> *plaqfield, Array<complexd> *poly, Array<complexd> *l2, Array<complexd> *lo2, size_t sl2, int radius, bool SquaredField, bool alongCharges, int perpPoint) : plaqfield(plaqfield), poly(poly), l2(l2), lo2(lo2), sl2(sl2), radius(radius), SquaredField(SquaredField), alongCharges(alongCharges), perpPoint(perpPoint) {
	size = SpatialVolume();
	timesec = 0.0;  
}
   ~L2ML(){ };
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
   void Run(){	return Run(0);}
   double flops(){	return ((double)flop() * 1.0e-9) / timesec;}
   double bandwidth(){	return (double)bytes() / (timesec * (double)(1 << 30));}
   long long flop() const { return 0;}
   long long bytes() const{ return 0;}
   double get_time(){	return timesec;}
   void stat(){	cout << "L2ML:  " <<  get_time() << " s\t"  << bandwidth() << " GB/s\t" << flops() << " GFlops"  << endl;}
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
  	l2->Backup();
  	lo2->Backup();	
  }
  void postTune() {  
	l2->Restore(); 
	lo2->Restore();
 }

};





__global__ void kernel_l2avg_l4_multilevel(complexd *dev_l2, complexd *dev_lo2, complexd *dev_l4, complexd *dev_lo4, double l2norm){

    size_t id = threadIdx.x + blockDim.x * blockIdx.x;
    size_t size = SpatialVolume() * (Dirs()-1);
    if(id >= size) return;
    
    
    //T4 and T04 calculus
	int nl2 = Grid(TDir())/2;
	int nl4 = Grid(TDir())/4;
	int l4 = 0;
	for(int l2 = 0; l2 < nl2; l2+=2){
		
		complexd t2[2];
		t2[0] = dev_l2[id + size * (l2 + 1)] * l2norm;
		t2[1] = dev_l2[id + size * l2] * l2norm;
		complexd pl0 = t2[1] * t2[0];
		dev_l4[id + size * l4] += pl0;
		
		for(int l0 = 0; l0 < 2; ++l0){
			int layer = l2 + l0;
						
			for(int tt = 0; tt < 2; ++tt)
			for(int iz = 0; iz < Grid(0); ++iz) {	
				int posin = id + size * layer + size * nl2 * tt + size * nl2 * 2 * iz;
				int posout = id + size * l4 + size * nl4 * l0 + size * nl4 * 2 * tt + size * nl4 * 4 * iz;
				for(int fi = 0; fi < 6; fi++){
					complexd pl = t2[l0] * dev_lo2[posin + size * nl4 * 4 * Grid(0) * fi] * l2norm;
					dev_lo4[posout + size * nl4 * 4 * Grid(0) * fi] += pl;
				}
			}
		}		
		l4++;	
	}
}


class L2AvgL4ML: Tunable{
private:
	Array<complexd> *l4;
	Array<complexd> *l2;
	Array<complexd> *lo2;
	Array<complexd> *lo4;
	double l2norm;
	size_t sl4;
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
	kernel_l2avg_l4_multilevel<<<tp.grid, tp.block, 0, stream>>>(l2->getPtr(), lo2->getPtr(), l4->getPtr(), lo4->getPtr(), l2norm);
}
public:	
   L2AvgL4ML(Array<complexd> *l2, Array<complexd> *lo2, Array<complexd> *l4, Array<complexd> *lo4, size_t sl4, double l2norm) : l2(l2),lo2(lo2), l4(l4), lo4(lo4), sl4(sl4), l2norm(l2norm) {
	size = SpatialVolume() * Grid(0) * (Dirs()-1);
	timesec = 0.0;  
}
   ~L2AvgL4ML(){ };
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
   void Run(){	return Run(0);}
   double flops(){	return ((double)flop() * 1.0e-9) / timesec;}
   double bandwidth(){	return (double)bytes() / (timesec * (double)(1 << 30));}
   long long flop() const { return 0;}
   long long bytes() const{ return 0;}
   double get_time(){	return timesec;}
   void stat(){	cout << "L2AvgL4ML:  " <<  get_time() << " s\t"  << bandwidth() << " GB/s\t" << flops() << " GFlops"  << endl;}
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
  	l4->Backup();
  	lo4->Backup();
  }
  void postTune() {  
  	l4->Restore(); 
  	lo4->Restore();
 }

};


__global__ void kernel_l4avg_Final_multilevel(complexd *dev_l4, complexd *dev_lo4, complexd *dev_pp, complexd *dev_ppo, double norm){
    size_t id = threadIdx.x + blockDim.x * blockIdx.x;    				
	
	int nl4 = Grid(TDir())/4;	
    //int size = SpatialVolume()
    int size = SpatialVolume() * (Dirs()-1);
	
	for(int fi = 0; fi < 6; fi++)
	for(int iz = 0; iz < Grid(0); ++iz) {
		complexd pp = 0.;
		complexd ppo = 0.;
		if( id < size ){			
			complexd pl = 1.;
			for(int l4 = 0; l4 < nl4; ++l4){			
				//int pos = id + size * dir + size * l4;		
				int pos = id + size * l4;
				complexd pl0 = dev_l4[pos] * norm;
				complexd pl1 = 1.;
				for(int l04 = 0; l04 < nl4; ++l04){
				 if(l4!=l04) pl1 *= dev_l4[id + size * l04] * norm;
				}
				pl *= pl0;
				 
				for(int t0 = 0; t0 < 2; ++t0)
				for(int t1 = 0; t1 < 2; ++t1){			
					int pos0 = pos + size * nl4 * t0 + size * nl4 * 2 * t1 + size * nl4 * 4 * iz;
					pos0 += size * nl4 * 4 * Grid(0) * fi;
					complexd pp0 = dev_lo4[pos0] * norm;
					ppo += pl1 * pp0;
				}
			}
			pp += pl;	
		}
		if(iz == 0 && fi == 0){
			reduce_block_1d<complexd>(dev_pp, pp);
			__syncthreads();
		}
		reduce_block_1d<complexd>(dev_ppo + iz + Grid(0) * fi, ppo);
		__syncthreads();
	}
}

class L4AvgPP: Tunable{
private:
	Array<complexd> *l4;
	Array<complexd> *lo4;
	Array<complexd> *dev_pp;
	Array<complexd> *dev_ppo;
	double norm_pp, norm_ppo;
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
	dev_pp->Clear();
	dev_ppo->Clear();
	kernel_l4avg_Final_multilevel<<<tp.grid, tp.block, tp.shared_bytes, stream>>>(l4->getPtr(), lo4->getPtr(), dev_pp->getPtr(), dev_ppo->getPtr(), l4norm);
}
public:
	Array<complexd> *pp;
	Array<complexd> *ppo;
	
   L4AvgPP(Array<complexd> *l4, Array<complexd> *lo4, double l4norm) : l4(l4), lo4(lo4), l4norm(l4norm) {
	size = SpatialVolume() * (Dirs()-1);
	dev_pp = new Array<complexd>(Device, 1);
	dev_ppo = new Array<complexd>(Device, Grid(0)*6);
	
	pp = new Array<complexd>(Host, 1);
	ppo = new Array<complexd>(Host, Grid(0)*6);
	norm_pp = 1. / double(SpatialVolume()*(Dirs()-1));
	norm_ppo = 1. / double(SpatialVolume()*(Dirs()-1)*4*Grid(TDir())/4);
	timesec = 0.0;  
}
   ~L4AvgPP(){ delete dev_pp; delete dev_ppo; };
   Array<complexd>* Run(const cudaStream_t &stream){
#ifdef TIMMINGS
    time.start();
#endif
	apply(stream);
	pp->Copy(dev_pp);
	ppo->Copy(dev_ppo);
	for(int i = 0; i < pp->Size(); ++i) pp->getPtr()[i] *= norm_pp;
	for(int i = 0; i < ppo->Size(); ++i) ppo->getPtr()[i] *= norm_ppo;
    cudaDevSync();
    cudaCheckError("Kernel execution failed");
#ifdef TIMMINGS
	cudaDevSync( );
    time.stop();
    timesec = time.getElapsedTimeInSec();
#endif
	return pp;
}
   Array<complexd>* Run(){	return Run(0);}
   double flops(){	return ((double)flop() * 1.0e-9) / timesec;}
   double bandwidth(){	return (double)bytes() / (timesec * (double)(1 << 30));}
   long long flop() const { return 0;}
   long long bytes() const{ return 0;}
   double get_time(){	return timesec;}
   void stat(){	cout << "L4AvgPP:  " <<  get_time() << " s\t"  << bandwidth() << " GB/s\t" << flops() << " GFlops"  << endl;}
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












__global__ void kernel_plaquette_comps(const double *lat, complexd* plaq_comps, complexd* mean_plaq){
    size_t idx = threadIdx.x + blockDim.x * blockIdx.x;   
	complexd plaq[6];
	for(int d=0; d<6; d++) plaq[d] = 0.0;			
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
		int pos = indexId(x);
		for(int d=0; d<6; d++) plaq_comps[pos + d * Volume()] = plaq[d];
	}
	for(int d=0; d<6; d++){
		reduce_block_1d<complexd>(mean_plaq + d, plaq[d]);
	  __syncthreads();
	}
}

	
class PlaqFields: Tunable{
public:
	Array<complexd>* fields;
	Array<complexd>* Meanfields;
	Array<complexd>* Meanfields_dev;
private:
	Array<double>* lat;
	int size;
	int sum;
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
	kernel_plaquette_comps<<<tp.grid, tp.block, tp.shared_bytes, stream>>>(lat->getPtr(), fields->getPtr(), Meanfields_dev->getPtr());
}
public:
   PlaqFields(Array<double>* lat) : lat(lat) {
    size = Volume();
   	fields = new Array<complexd>(Device, 6*size );
   	Meanfields_dev = new Array<complexd>(Device, 6 );
   	Meanfields_dev->Clear();
   	Meanfields = new Array<complexd>(Host, 6 );
	timesec = 0.0;  
	sum = 0;
}
   ~PlaqFields(){ delete fields; delete Meanfields_dev; delete Meanfields; };
   Array<complexd>* Run(const cudaStream_t &stream){
#ifdef TIMMINGS
    time.start();
#endif
	apply(stream);
	sum++;
    cudaDevSync();
    cudaCheckError("Kernel execution failed");
#ifdef TIMMINGS
	cudaDevSync( );
    time.stop();
    timesec = time.getElapsedTimeInSec();
#endif
	return fields;
}
   Array<complexd>* Run(){ return Run(0); }
   
   Array<complexd>* getPlaqField(){ return fields; }
   
   Array<complexd>* GetMean(){
		Meanfields->Copy(Meanfields_dev);
		for(int i = 0; i < Meanfields->Size(); i++)
			Meanfields->at(i) /= double(size*sum);	   	
	   return Meanfields;
   }
   
   double flops(){	return ((double)flop() * 1.0e-9) / timesec;}
   double bandwidth(){	return (double)bytes() / (timesec * (double)(1 << 30));}
   long long flop() const { return 0;}
   long long bytes() const{ return 0;}
   double get_time(){	return timesec;}
   void stat(){	cout << "PlaqFields:  " <<  get_time() << " s\t"  << bandwidth() << " GB/s\t" << flops() << " GFlops"  << endl;}
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
  void preTune() { Meanfields_dev->Backup(); }
  void postTune() { Meanfields_dev->Restore(); }

};






Array<complexd>* MultiLevelTTO(Array<double> *lat, CudaRNG *rng_state, int n4, int k4, int n2, int k2, int metrop, int ovrn, int radius, bool SquaredField, bool alongCharges, bool symmetrize, int perpPoint){
	Timer a0; a0.start();

	cout << "R: " << radius << endl;
	cout << "Level 0:" << endl;
	cout << "\tNº time links per slice: " << 2 << endl;
	cout << "\tNº iterations: " << n2 << endl;
	cout << "\tNº updates: " << k2 << endl;
	cout << "\tNº metropolis updates: " << metrop << endl;
	cout << "\tNº overrelaxation updates: " << ovrn << endl;
	
	cout << "Level 1:" << endl;
	cout << "\tNº time links per slice: " << 4 << endl;
	cout << "\tNº iterations: " << n4 << endl;
	cout << "\tNº updates: " << k4 << endl;
	cout << "\tNº metropolis updates: " << metrop << endl;
	cout << "\tNº overrelaxation updates: " << ovrn << endl;
	
	if(Dirs() < 4){
		cout << "Only implemented for 4D lattice..." << endl;
		Finalize(1);
	}

	if( Grid(TDir())%4 != 0 ) {
		cout << "Error: Cannot Apply MultiLevel Algorithm...\n Nt is not multiple of 4...\n Exiting..." << endl;
		exit(1);
	}
	
	if(perpPoint > Grid(0)/2-1 || perpPoint < -Grid(0)/2){
		cout << "Perpendicular point (" << perpPoint << ") should be between [" << -Grid(0)/2 << ":" << Grid(0)/2-1 << "]" << endl;
		exit(0);
	}
	
	
	
	Array<double>* dev_lat = new Array<double>(Device);
	dev_lat->Copy(lat);

	int nl2 = Grid(TDir())/2;
	int sl2 = nl2*(Dirs()-1)*SpatialVolume();
	Array<complexd> *l2 = new Array<complexd>(Device, sl2);
	int slo2 = (Dirs()-1)*Grid(0)*Grid(0)*6*SpatialVolume();
	Array<complexd> *lo2 = new Array<complexd>(Device, slo2);
	int nl4 = Grid(TDir())/4;
	size_t sl4 = nl4*(Dirs()-1)*SpatialVolume();
	Array<complexd> *l4 = new Array<complexd>(Device, sl4);
	size_t slo4 = Grid(0)*6*(Dirs()-1)*Grid(0)*SpatialVolume();
	Array<complexd> *lo4 = new Array<complexd>(Device, slo4);
	
	// metropolis and overrelaxation algorithm
	Metropolis_ML<4> mtp4(dev_lat, rng_state, metrop);
	OverRelaxation_ML<4> ovr4(dev_lat, ovrn);
	
	Metropolis_ML<2> mtp2(dev_lat, rng_state, metrop);
	OverRelaxation_ML<2> ovr2(dev_lat, ovrn);
	
	const bool multihit = true;
	Polyakov_Volume<multihit> mhitVol(dev_lat);
	Array<complexd>* dev_mhit;
	
	double l2norm = 1./double(n2);
	L2AvgL4ML l2avgl4(l2, lo2, l4, lo4, sl4, l2norm);
	double l4norm = 1./double(n4);
	L4AvgPP l4avgpp(l4,lo4, l4norm);
	
	PlaqFields plaqf(dev_lat);
	
	l4->Clear();
	lo4->Clear();
	for(int i = 0; i < n4; ++i){
		cout << "Iter of l4: " << i << endl;
		//Update the lattice k4 times freezing spacial links in layers with t multiple of 4
		for(int j = 0; j < k4; ++j){
			mtp4.Run();
			ovr4.Run();
		}
		l2->Clear();
		lo2->Clear();
		for(int k = 0; k < n2; ++k){		
			//Update the lattice k2 times freezing spacial links in layers with t multiple of 2
			for(int l = 0; l < k2; ++l){
				mtp2.Run();
				ovr2.Run();	
			}
			//Extract plaquette components and mean plaquette components
			Array<complexd>* plaqfield = plaqf.Run();
			//Extract temporal links and apply MultiHit
			dev_mhit = mhitVol.Run();			
			//Calculate tensor T2
			L2ML l2ml(plaqfield, dev_mhit, l2, lo2, sl2, radius, SquaredField, alongCharges, perpPoint);
			l2ml.Run();
		}
		//Average tensor T2 and Calculate tensor T4
		l2avgl4.Run();	
		
		
		if(0){
			double l4norm1 = 1./double(i+1);
			L4AvgPP l4avgpp1(l4, lo4, l4norm1);
			Array<complexd>* res = l4avgpp1.Run();
			cout << res << endl;
			delete res;
		}
	}
	delete dev_lat;
	delete dev_mhit;
	delete l2;
	delete lo2;
	//Average tensor T4 and Calculate P(0)*conj(P(r))	
	Array<complexd>* pp = l4avgpp.Run();
	Array<complexd>* ppo = l4avgpp.ppo;
	delete l4;
	delete lo4;

	std::ofstream fileout;
	std::string filename = "Pot_mlevel_TTO_" + GetLatticeNameI();
	filename += "_" + ToString(n4) + "_" + ToString(k4);
	filename += "_" + ToString(n2) + "_" + ToString(k2);
	filename += "_" + ToString(metrop) + "_" + ToString(ovrn);
	filename += "_radius_" + ToString(radius);
	filename += "_pP_" + ToString(perpPoint);
	if(SquaredField) filename += "_squared";
	if(alongCharges) filename += "_chargeplane";
	if(symmetrize) filename += "_sym";
	filename += ".dat";
	fileout.open (filename.c_str());
	if (!fileout.is_open()) {
		std::cout << "Error opening file: " << filename << std::endl;
		exit(1);
	}
	cout << "Saving data to " << filename << endl;
	fileout << std::scientific;
	fileout.precision(14);
	
	
	if(symmetrize){	
		for(int fi = 0; fi < 1; ++fi)
		for(int iz = 1; iz < Grid(0)/2; ++iz){
			int rp = iz + Grid(0)/2 + fi * Grid(0);
			int rm = -iz + Grid(0)/2 + fi * Grid(0);
			//cout << rm << '\t' << rm-Grid(0)/2 << '\t' << rp << '\t' << rp-Grid(0)/2 << endl;
			complexd val = ( ppo->at(rm) + ppo->at(rm) ) * 0.5;
			ppo->at(rm) = val;
			ppo->at(rp) = val;
		}	
	}
		
	cout << radius << '\t' << pp->at(0) << endl;
	fileout << radius << '\t' << pp->at(0) << endl;	
	Array<complexd>* splaq = plaqf.GetMean();
	for(int fi = 0; fi < 6; ++fi){
		cout << splaq->at(fi) << endl;
		fileout << splaq->at(fi) << endl;
	}	
	fileout << Grid(0) << endl;
	for(int iz = 0; iz < Grid(0); ++iz){
		//cout << r << '\t' << r - Grid(0)/2;
		cout << iz - Grid(0)/2;
		fileout << iz - Grid(0)/2;
		for(int fi = 0; fi < 6; ++fi) {
			int id = iz + Grid(0) * fi;
			if(SquaredField) cout << '\t' << ppo->at(id) << '\t' << ppo->at(id) / pp->at(0)-splaq->at(fi).real();
			else cout << '\t' << ppo->at(id) << '\t' << ppo->at(id) / pp->at(0)-splaq->at(fi).imag();
			fileout << '\t' << ppo->at(id);// << '\t' << ppo->at(id) / pp->at(0);
		}
		cout << endl;
		fileout << endl;
	}
	fileout.close();
	delete ppo;	
	a0.stop();
	std::cout << "time " << a0.getElapsedTime() << " s" << endl;
	return pp;
}

}

Array<complexd>* MultiLevelTTO(Array<double> *lat, CudaRNG *rng_state, int n4, int k4, int n2, int k2, int metrop, int ovrn, int radius, bool SquaredField, bool alongCharges, bool symmetrize, int perpPoint){
	return ML_TTO::MultiLevelTTO(lat, rng_state, n4, k4, n2, k2, metrop, ovrn, radius, SquaredField, alongCharges, symmetrize, perpPoint);
}

}
