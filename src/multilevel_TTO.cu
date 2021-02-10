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
#include "multilevel_TTO.cuh"



__global__ void kernel_l2_multilevel_1(complexd *plaqfield, complexd *poly, complexd *l2, complexd *lo2, int radius, int nl0, bool SquaredField, bool alongCharges, int2 perpPoint){
    uint id = threadIdx.x + blockDim.x * blockIdx.x;    
	if(id >= SpatialVolume()) return;		
	int x[4];
	indexNOSD(id, x);
	
	int nl2 = Grid(TDir())/nl0;
	for(int dir = 0; dir < TDir(); dir++){		
		int layer = 0;
		for(int t = 0; t < Grid(TDir()); t+=nl0){
			complexd pl0 = 1.;
			complexd pl1 = 1.;
			for(x[TDir()] = t; x[TDir()] < t+nl0; ++x[TDir()]){
				pl0 *= poly[indexId(x)];
				int xold = x[dir];
				x[dir] = (x[dir] + radius) % Grid(dir);
				pl1 *= conj(poly[indexId(x)]);								
				x[dir] = xold;
			}			
			complexd pl = pl0 * pl1;
			
			for(int tt = 0; tt < nl0; ++tt)
			for(int iz = 0; iz < Grid(0); ++iz){	
				complexd plaq = 1.0;
				x[TDir()] = t + tt;				
				
				int rad = (radius)/2;				
				int evenradius = (radius+1)%2;	
				//int evenradius = (radius+1) & 1;
								
				int dirz = dir;				//Z
				int dirx = (dirz+1)%TDir();  //X
				int diry = (dirx+1)%TDir(); //Y		
				
				int pos = 0;
				complexd field[6];
				if(alongCharges){
					int xoldx = x[dirx];
					int xoldy = x[diry];
					int xoldz = x[dirz];
					//x[dirz] = (x[dirz] + rad + iz - Grid(dirz)/2 + Grid(dirz)) % Grid(dirz);							
					x[dirz] = (x[dirz] + rad) % Grid(dirz);				
					x[dirz] = (x[dirz] + iz - Grid(dirz)/2 + Grid(dirz)) % Grid(dirz);												
					x[dirx] = (x[dirx] + perpPoint.x + Grid(dirx)) % Grid(dirx);
					x[diry] = (x[diry] + perpPoint.y + Grid(diry)) % Grid(diry);
					pos = indexId(x); 	
					x[dirx] = xoldx;
					x[diry] = xoldy;
					x[dirz] = xoldz;
				}
				else{
					int xoldx = x[dirx];
					int xoldy = x[diry];
					int xoldz = x[dirz];
					x[dirz] = (x[dirz] + rad + perpPoint.y + Grid(dirz)) % Grid(dirz);					
					x[dirx] = (x[dirx] + iz - Grid(dirx)/2 + Grid(dirx)) % Grid(dirx);
					x[diry] = (x[diry] + perpPoint.x + Grid(diry)) % Grid(diry);					
					pos = indexId(x); 	
					x[dirx] = xoldx;
					x[diry] = xoldy;
					x[dirz] = xoldz;
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
				idout +=  SpatialVolume() * (Dirs()-1) * nl2 * tt + SpatialVolume() * (Dirs()-1) * nl2 * nl0 * iz;
				for(int fi = 0; fi < 6; fi++){
					field[fi] = pl * field[fi];
					int idout1 = idout + SpatialVolume() * (Dirs()-1) * nl2 * nl0 * Grid(0) * fi;
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
	int nl0;
	int radius;
	int size;
	int2 perpPoint;
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
	kernel_l2_multilevel_1<<<tp.grid, tp.block, tp.shared_bytes, stream>>>(plaqfield->getPtr(), poly->getPtr(), l2->getPtr(), lo2->getPtr(), radius, nl0, SquaredField, alongCharges, perpPoint);
}
public:	
   L2ML(Array<complexd> *plaqfield, Array<complexd> *poly, Array<complexd> *l2, Array<complexd> *lo2, size_t sl2, int radius, int nl0, bool SquaredField, bool alongCharges, int2 perpPoint) : plaqfield(plaqfield), poly(poly), l2(l2), lo2(lo2), sl2(sl2), radius(radius), nl0(nl0), SquaredField(SquaredField), alongCharges(alongCharges), perpPoint(perpPoint) {
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





__global__ void kernel_l2avg_l4_multilevel(complexd *dev_l2, complexd *dev_lo2, complexd *dev_l4, complexd *dev_lo4, double l2norm, int nl0, int nl1){

    size_t id = threadIdx.x + blockDim.x * blockIdx.x;
    size_t size = SpatialVolume() * (Dirs()-1);
    if(id >= size) return;
    
    
    //T4 and T04 calculus
	int nl2 = Grid(TDir())/nl0;
	int nl4 = Grid(TDir())/nl1;
	int l1 = nl1/nl0;
	int l4 = 0;
	for(int l2 = 0; l2 < nl2; l2+=l1){
		
		complexd pl0 = 1.;
		for(int l0 = 0; l0 < l1; ++l0){
			int layer = l2 + l0;
			pl0 *= dev_l2[id + size * layer] * l2norm;
		}
		dev_l4[id + size * l4] += pl0;
		
		for(int l0 = 0; l0 < l1; ++l0){
			int layer = l2 + l0;
			
			
			complexd pl1 = 1.;
			for(int l00 = 0; l00 < l1; ++l00){
			 if(l0!=l00) pl1 *= dev_l2[id + size * (l2 + l00)] * l2norm;
			}
			
			for(int tt = 0; tt < nl0; ++tt)
			for(int iz = 0; iz < Grid(0); ++iz) {	
				//NEED TO CHECK THIS PART!!!!!!!!
				int posin = id + size * layer + size * nl2 * tt + size * nl2 * nl0 * iz;
				int posout = id + size * l4 + size * nl4 * l0 + size * nl4 * nl0 * tt + size * nl4 * nl0 * l1 * iz;
				for(int fi = 0; fi < 6; fi++){
					complexd pl = pl1 * dev_lo2[posin + size * nl4 * nl0 * l1 * Grid(0) * fi] * l2norm;
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
	int nl0;
	int nl1;
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
	kernel_l2avg_l4_multilevel<<<tp.grid, tp.block, 0, stream>>>(l2->getPtr(), lo2->getPtr(), l4->getPtr(), lo4->getPtr(), l2norm, nl0, nl1);
}
public:	
   L2AvgL4ML(Array<complexd> *l2, Array<complexd> *lo2, Array<complexd> *l4, Array<complexd> *lo4, size_t sl4, double l2norm, int nl0, int nl1) : l2(l2),lo2(lo2), l4(l4), lo4(lo4), sl4(sl4), l2norm(l2norm), nl0(nl0), nl1(nl1) {
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


__global__ void kernel_l4avg_Final_multilevel(complexd *dev_l4, complexd *dev_lo4, complexd *dev_pp, complexd *dev_ppo, double norm, int nl0, int nl1){
    size_t id = threadIdx.x + blockDim.x * blockIdx.x;    				
	
	int nl4 = Grid(TDir())/nl1;	
    int size = SpatialVolume() * (Dirs()-1);
    
	int l1 = nl1/nl0;
	
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
				 
				for(int t0 = 0; t0 < nl0; ++t0)
				for(int t1 = 0; t1 < l1; ++t1){			
					int pos0 = pos + size * nl4 * t0 + size * nl4 * nl0 * t1 + size * nl4 * nl0 * l1 * iz;
					pos0 += size * nl4 * nl0 * l1 * Grid(0) * fi;
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
	int nl0;
	int nl1;
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
	kernel_l4avg_Final_multilevel<<<tp.grid, tp.block, tp.shared_bytes, stream>>>(l4->getPtr(), lo4->getPtr(), dev_pp->getPtr(), dev_ppo->getPtr(), l4norm, nl0, nl1);
}
public:
	Array<complexd> *pp;
	Array<complexd> *ppo;
	
   L4AvgPP(Array<complexd> *l4, Array<complexd> *lo4, double l4norm, int nl0, int nl1) : l4(l4), lo4(lo4), l4norm(l4norm), nl1(nl1), nl0(nl0) {
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








ML_Fields* MultiLevelTTO(Array<double> *lat, CudaRNG *rng_state, MLTTOArg *arg){
//, int nl0, int nl1, int n4, int k4, int n2, int k2, int metrop, int ovrn, int radius, bool SquaredField, bool alongCharges, bool symmetrize, int2 perpPoint, bool ppmhit, bool plaqmhit){
	Timer a0; a0.start();
	
	arg->check();	
	
	Array<double>* dev_lat = new Array<double>(Device);
	dev_lat->Copy(lat);

	int nl2 = Grid(TDir())/arg->nLinksLvl0();
	int sl2 = nl2*(Dirs()-1)*SpatialVolume();
	Array<complexd> *l2 = new Array<complexd>(Device, sl2);
	int slo2 = (Dirs()-1)*Grid(0)*Grid(0)*6*SpatialVolume();
	Array<complexd> *lo2 = new Array<complexd>(Device, slo2);
	int nl4 = Grid(TDir())/arg->nLinksLvl1();
	size_t sl4 = nl4*(Dirs()-1)*SpatialVolume();
	Array<complexd> *l4 = new Array<complexd>(Device, sl4);
	size_t slo4 = Grid(0)*6*(Dirs()-1)*Grid(0)*SpatialVolume();
	Array<complexd> *lo4 = new Array<complexd>(Device, slo4);
	
	
	// metropolis and overrelaxation algorithm
	Metropolis_ML mtp(dev_lat, rng_state);
	OverRelaxation_ML ovr(dev_lat);
	
	double l2norm = 1./double(arg->StepsLvl0());
	L2AvgL4ML l2avgl4(l2, lo2, l4, lo4, sl4, l2norm, arg->nLinksLvl0(), arg->nLinksLvl1());
	double l4norm = 1./double(arg->StepsLvl1());
	L4AvgPP l4avgpp(l4, lo4, l4norm, arg->nLinksLvl0(), arg->nLinksLvl1());
	
	
	
	
	Polyakov_Volume0 mhitVol(dev_lat, arg->PPMHit(), arg->PlaqMHit());
	Array<complexd>* latmhit = mhitVol.GetLatMHit();
	Array<complexd>* dev_mhit = mhitVol.GetPolyVol();
	
	
	PlaqFields<double> plaqf(dev_lat);
	PlaqFields<complexd> plaqfmhit(latmhit);
	
	
	
	
	
	
	
	
	l4->Clear();
	lo4->Clear();
	for(int i = 0; i < arg->StepsLvl1(); ++i){
		cout << "Iter of l4: " << i << endl;
		//Update the lattice k4 times freezing spacial links in layers with t multiple of 4
		for(int j = 0; j < arg->UpdatesLvl1(); ++j){
			mtp.Run(arg->nUpdatesMetropolis(), arg->nLinksLvl1());
			ovr.Run(arg->nUpdatesOvr(), arg->nLinksLvl1());
		}
		l2->Clear();
		lo2->Clear();
		for(int k = 0; k < arg->StepsLvl0(); ++k){		
			//Update the lattice k2 times freezing spacial links in layers with t multiple of 2
			for(int l = 0; l < arg->UpdatesLvl0(); ++l){
				mtp.Run(arg->nUpdatesMetropolis(), arg->nLinksLvl0());
				ovr.Run(arg->nUpdatesOvr(), arg->nLinksLvl0());	
			}
			//Extract temporal links and apply MultiHit
			mhitVol.Run();
			//Extract plaquette components and mean plaquette components
			Array<complexd>* plaqfield;
			if(arg->PlaqMHit()) plaqfield = plaqfmhit.Run();
			else plaqfield = plaqf.Run();			
			//Calculate tensor T2
			L2ML l2ml(plaqfield, dev_mhit, l2, lo2, sl2, arg->Radius(), arg->nLinksLvl0(), arg->SquaredField(), arg->AlongCharges(), arg->PerpPoint());
			l2ml.Run();
		}
		//Average tensor T2 and Calculate tensor T4
		l2avgl4.Run();	
		
		
		if(0){
			double l4norm1 = 1./double(i+1);
			L4AvgPP l4avgpp1(l4, lo4, l4norm1, arg->nLinksLvl0(), arg->nLinksLvl1());
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
	
	filename += "_" + ToString(arg->nLinksLvl1()) + "_" + ToString(arg->StepsLvl1());
	filename += "_" + ToString(arg->UpdatesLvl1()) + "_" + ToString(arg->nLinksLvl0());
	filename += "_" + ToString(arg->StepsLvl0()) + "_" + ToString(arg->UpdatesLvl0());
	filename += "_" + ToString(arg->nUpdatesMetropolis()) + "_" + ToString(arg->nUpdatesOvr());
	filename += "_radius_" + ToString(arg->Radius());
	if(arg->AlongCharges()) filename += "_p(" + ToString(arg->PerpPoint().x) + "," + ToString(arg->PerpPoint().y) + ",z)";
	else filename += "_p(x, " + ToString(arg->PerpPoint().x) + "," + ToString(arg->PerpPoint().y) + ")";
	if(arg->SquaredField()) filename += "_squared";
	if(arg->AlongCharges()) filename += "_chargeplane";
	if(arg->Sym()) filename += "_sym";
	filename += ".dat";
	fileout.open (filename.c_str());
	if (!fileout.is_open()) {
		std::cout << "Error opening file: " << filename << std::endl;
		exit(1);
	}
	cout << "Saving data to " << filename << endl;
	fileout << std::scientific;
	fileout.precision(14);
	
	
	if(arg->Sym()){	
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
		
	cout << arg->Radius() << '\t' << pp->at(0) << endl;
	fileout << arg->Radius() << '\t' << pp->at(0) << endl;	
	Array<complexd>* splaq;
	if(arg->PlaqMHit()) splaq = plaqfmhit.GetMean();
	else splaq = plaqf.GetMean();
	for(int fi = 0; fi < 6; ++fi){
		//cout << splaq->at(fi) << endl;
		fileout << splaq->at(fi) << endl;
	}	
	fileout << Grid(0) << endl;
	for(int iz = 0; iz < Grid(0); ++iz){
		//cout << r << '\t' << r - Grid(0)/2;
		cout << iz - Grid(0)/2;
		fileout << iz - Grid(0)/2;
		for(int fi = 0; fi < 6; ++fi) {
			int id = iz + Grid(0) * fi;
			//if(arg->SquaredField()) cout << '\t' << ppo->at(id) << '\t' << ppo->at(id) / pp->at(0)-splaq->at(fi).real();
			//else cout << '\t' << ppo->at(id) << '\t' << ppo->at(id) / pp->at(0)-splaq->at(fi).imag();
			fileout << '\t' << ppo->at(id);// << '\t' << ppo->at(id) / pp->at(0);
		}
		//cout << endl;
		fileout << endl;
	}
	fileout.close();
	
	
	
	for(int iz = 0; iz < Grid(0); ++iz){
		//cout << iz - Grid(0)/2;
		//fileout << iz - Grid(0)/2;
		for(int fi = 0; fi < 6; ++fi) {
			int id = iz + Grid(0) * fi;
			if(arg->SquaredField()) ppo->at(id) = ppo->at(id) / pp->at(0)-splaq->at(fi).real();
			else ppo->at(id) = ppo->at(id) / pp->at(0)-splaq->at(fi).imag();
		}
	}
	ML_Fields *data = new ML_Fields;
	data->Set(splaq, pp, ppo);

	a0.stop();
	std::cout << "time " << a0.getElapsedTime() << " s" << endl;
	return data;
}
}

ML_Fields* MultiLevelTTO(Array<double> *lat, CudaRNG *rng_state, MLTTOArg *arg){
	return ML_TTO::MultiLevelTTO(lat, rng_state, arg);
}



}
