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


#include "multilevel_common.cuh"





template<bool multihit>
__global__ void kernel_l2_multilevel_11(double *lat, complexd *l2, int Rmax, int nl0){
    size_t id = threadIdx.x + blockDim.x * blockIdx.x;    
	if(id >= SpatialVolume()) return;	
	
	int parity = 0;
	if(id >= SpatialVolume()/2){
		id -= SpatialVolume()/2;
		parity = 1;
	} 	
	int x[4];
	indexEO(id, parity, x);
	
	int nlayers = Grid(TDir())/nl0;
	for(int r = 0; r < Rmax; ++r){	
		for(int dir = 0; dir < TDir(); dir++){		
			int layer = 0;
			for(int t = 0; t < Grid(TDir()); t+=nl0){
				complexd pl0 = 1.;
				complexd pl1 = 1.;
				for(x[TDir()] = t; x[TDir()] < t+nl0; ++x[TDir()]){
					int newid = indexId(x) >> 1;
					int parity = GetParity(x);
					if(multihit){
						pl0 *= MultiHit(lat, newid, parity, TDir());
					}
					else{
						pl0 *= exp_ir(lat[newid + parity * HalfVolume() + TDir() * Volume()]);
					}
					int xold = x[dir];
					x[dir] = (x[dir] + r) % Grid(dir);
					newid = indexId(x) >> 1;
					parity = GetParity(x);
					if(multihit){
						pl1 *= conj(MultiHit(lat, newid, parity, TDir()));
					}
					else{
						pl1 *= conj(exp_ir(lat[newid + parity * HalfVolume() + TDir() * Volume()]));
					}
									
					x[dir] = xold;
				}			
				complexd pl= pl0 * pl1;			
				int pos = indexIdS(x) + SpatialVolume() * r + SpatialVolume() * Rmax * dir + SpatialVolume() * Rmax * (Dirs()-1) * layer;
				l2[pos] = pl + l2[pos];
				layer++;
			}
		}
	}
}


template<bool multihit>
class L2ML1: Tunable{
private:
	Array<double> *lat;
	Array<complexd> *l2;
	int nl0;
	size_t sl2;
	int Rmax;
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
	kernel_l2_multilevel_11<multihit><<<tp.grid, tp.block, tp.shared_bytes, stream>>>(lat->getPtr(), l2->getPtr(), Rmax, nl0);
}
public:	
   L2ML1(Array<double> *lat, Array<complexd> *l2, size_t sl2, int Rmax, int nl0) : lat(lat), l2(l2), sl2(sl2), Rmax(Rmax), nl0(nl0) {
	size = SpatialVolume();
	timesec = 0.0;  
}
   ~L2ML1(){ };
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
   void stat(){	cout << "kernel_l2_multilevel_1:  " <<  get_time() << " s\t"  << bandwidth() << " GB/s\t" << flops() << " GFlops"  << endl;}
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
  }
  void postTune() {  
	l2->Restore();
 }

};
























__global__ void kernel_l2_multilevel_1(complexd *poly, complexd *l2, int Rmax, int nl0){
    size_t id = threadIdx.x + blockDim.x * blockIdx.x;    
	if(id >= SpatialVolume()) return;		
	int x[4];
	indexNOSD(id, x);
	
	int nlayers = Grid(TDir())/nl0;
	for(int r = 0; r < Rmax; ++r){	
		for(int dir = 0; dir < TDir(); dir++){		
			int layer = 0;
			for(int t = 0; t < Grid(TDir()); t+=nl0){
				complexd pl0 = 1.;
				complexd pl1 = 1.;
				for(x[TDir()] = t; x[TDir()] < t+nl0; ++x[TDir()]){
					pl0 *= (poly[indexId(x)]);
					int xold = x[dir];
					x[dir] = (x[dir] + r) % Grid(dir);
					pl1 *= conj(poly[indexId(x)]);
									
					x[dir] = xold;
				}			
				complexd pl= pl0 * pl1;			
				int pos = id + SpatialVolume() * r + SpatialVolume() * Rmax * dir + SpatialVolume() * Rmax * (Dirs()-1) * layer;
				l2[pos] = pl + l2[pos];
				layer++;
			}
		}
	}
}










class L2ML: Tunable{
private:
	Array<complexd> *poly;
	Array<complexd> *l2;
	int nl0;
	size_t sl2;
	int Rmax;
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
	kernel_l2_multilevel_1<<<tp.grid, tp.block, tp.shared_bytes, stream>>>(poly->getPtr(), l2->getPtr(), Rmax, nl0);
}
public:	
   L2ML(Array<complexd> *poly, Array<complexd> *l2, size_t sl2, int Rmax, int nl0) : poly(poly), l2(l2), sl2(sl2), Rmax(Rmax), nl0(nl0) {
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
   void stat(){	cout << "kernel_l2_multilevel_1:  " <<  get_time() << " s\t"  << bandwidth() << " GB/s\t" << flops() << " GFlops"  << endl;}
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
  }
  void postTune() {  
	l2->Restore();
 }

};





__global__ void kernel_l2avg_l4_multilevel(complexd *dev_l2, complexd *dev_l4, int Rmax, double l2norm, int nl0, int nl1){
    size_t id = threadIdx.x + blockDim.x * blockIdx.x;
    size_t size = SpatialVolume() * Rmax * (Dirs()-1);
    if(id >= size) return;			
	
	int nl2 = Grid(TDir())/nl0;
	int l1 = nl1/nl0;
	int l4 = 0;
	for(int l2 = 0; l2 < nl2; l2+=l1){
		complexd pl = 1.;
		for(int layer = l2; layer < l2+l1; ++layer){
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
	Array<complexd> *l4;
	Array<complexd> *l2;
	int nl0, nl1;
	double l2norm;
	size_t sl4;
	int Rmax;
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
	kernel_l2avg_l4_multilevel<<<tp.grid, tp.block, 0, stream>>>(l2->getPtr(), l4->getPtr(), Rmax, l2norm, nl0, nl1);
}
public:	
   L2AvgL4ML(Array<complexd> *l2, Array<complexd> *l4, size_t sl4, int Rmax, double l2norm, int nl0, int nl1) : l2(l2), l4(l4), sl4(sl4), Rmax(Rmax), l2norm(l2norm), nl0(nl0), nl1(nl1) {
	size = SpatialVolume() * Rmax * (Dirs()-1);
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
   void stat(){	cout << "kernel_l2avg_l4_multilevel:  " <<  get_time() << " s\t"  << bandwidth() << " GB/s\t" << flops() << " GFlops"  << endl;}
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
  }
  void postTune() {  
  	l4->Restore();
 }

};


template<bool savePPspace>
__global__ void kernel_l4avg_Final_multilevel(complexd *dev_l4, complexd *res, complexd *ppSpace, int Rmax, double norm, int nl1){
    size_t id = threadIdx.x + blockDim.x * blockIdx.x;    				
	
	int nl4 = Grid(TDir())/nl1;	
	for(int r = 0; r < Rmax; ++r)	{
		complexd pp = 0.;
		if( id < SpatialVolume() ){
			for(int dir = 0; dir < TDir(); dir++){
				complexd pl = 1.;
				for(int l4 = 0; l4 < nl4; ++l4){
					int newid = id + SpatialVolume() * r + SpatialVolume() * Rmax * dir + SpatialVolume() * Rmax * (Dirs()-1) * l4;
					pl *= dev_l4[newid] * norm;
				}
				pp += pl;
				if(savePPspace) ppSpace[id + SpatialVolume() * dir + SpatialVolume() * (Dirs()-1) * r] = pl;
			}
		}
		reduce_block_1d<complexd>(res + r, pp);
		__syncthreads();
	}
}

template<bool savePPspace>
class L4AvgPP: Tunable{
private:
	Array<complexd> *l4;
	Array<complexd> *dev_poly;
	Array<complexd> *poly;
	int nl1;
	int Rmax;
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
	dev_poly->Clear();
	if(savePPspace) kernel_l4avg_Final_multilevel<savePPspace><<<tp.grid, tp.block, tp.shared_bytes, stream>>>(l4->getPtr(), dev_poly->getPtr(), ppSpace->getPtr(), Rmax, l4norm, nl1);
	else kernel_l4avg_Final_multilevel<savePPspace><<<tp.grid, tp.block, tp.shared_bytes, stream>>>(l4->getPtr(), dev_poly->getPtr(), 0, Rmax, l4norm, nl1);
}
public:
	Array<complexd> *ppSpace = 0;
	Array<complexd>* getField(){ return ppSpace; }
	
   L4AvgPP(Array<complexd> *l4, int Rmax, double l4norm, int nl1) : l4(l4), Rmax(Rmax), l4norm(l4norm), nl1(nl1) {
	size = SpatialVolume();
	dev_poly = new Array<complexd>(Device, Rmax);
	if(savePPspace) ppSpace = new Array<complexd>(Device, SpatialVolume() * Rmax * (Dirs()-1));
	poly = new Array<complexd>(Host, Rmax);
	norm = 1. / double(SpatialVolume()*(Dirs()-1));
	timesec = 0.0;  
}
   ~L4AvgPP(){ delete dev_poly; };
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
   void stat(){	cout << "kernel_l4avg_Final_multilevel:  " <<  get_time() << " s\t"  << bandwidth() << " GB/s\t" << flops() << " GFlops"  << endl;}
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







Array<complexd>* MultiLevel(Array<double> *lat, CudaRNG *rng_state, MLArg *arg, bool PrintResultsAtEveryN4){
	Timer a0; a0.start();
	
	arg->check();	
	arg->print();
	
	
	Array<double>* dev_lat = new Array<double>(Device);
	dev_lat->Copy(lat);

	int nl2 = Grid(TDir())/arg->nLinksLvl0();
	int sl2 = nl2*(Dirs()-1)*arg->Rmax()*SpatialVolume();
	Array<complexd> *l2 = new Array<complexd>(Device, sl2);
	int nl4 = Grid(TDir())/arg->nLinksLvl1();
	size_t sl4 = nl4*(Dirs()-1)*arg->Rmax()*SpatialVolume();
	Array<complexd> *l4 = new Array<complexd>(Device, sl4);
	
	
	// metropolis and overrelaxation algorithm
	Metropolis_ML mtp(dev_lat, rng_state);
	OverRelaxation_ML ovr(dev_lat);
		
	const bool multihit = true;
	Polyakov_Volume<multihit> mhitVol(dev_lat);
	Array<complexd>* dev_mhit;
	
	double l2norm = 1./double(arg->StepsLvl0());
	L2AvgL4ML l2avgl4(l2, l4, sl4, arg->Rmax(), l2norm, arg->nLinksLvl0(), arg->nLinksLvl1());
	double l4norm = 1./double(arg->StepsLvl1());
	L4AvgPP<false> l4avgpp(l4, arg->Rmax(), l4norm, arg->nLinksLvl1());

	l4->Clear();
	for(int i = 0; i < arg->StepsLvl1(); ++i){
		cout << "Iter of l4: " << i << endl;
		//Update the lattice k4 times freezing spacial links in layers with t multiple of 4
		for(int j = 0; j < arg->UpdatesLvl1(); ++j){
			mtp.Run(arg->nUpdatesMetropolis(), arg->nLinksLvl1());
			ovr.Run(arg->nUpdatesOvr(), arg->nLinksLvl1());
		}
		l2->Clear();
		for(int k = 0; k < arg->StepsLvl0(); ++k){		
			//Update the lattice k2 times freezing spacial links in layers with t multiple of 2
			for(int l = 0; l < arg->UpdatesLvl0(); ++l){
				mtp.Run(arg->nUpdatesMetropolis(), arg->nLinksLvl0());
				ovr.Run(arg->nUpdatesOvr(), arg->nLinksLvl0());
			}
			//Extract temporal links and apply MultiHit
			dev_mhit = mhitVol.Run();			
			//Calculate tensor T2
			L2ML l2ml(dev_mhit, l2, sl2, arg->Rmax(), arg->nLinksLvl0());
			//L2ML1<multihit> l2ml(dev_lat, l2, sl2, arg->Rmax(), arg->nLinksLvl0());  // <--- SLOW
			l2ml.Run();
		}
		//Average tensor T2 and Calculate tensor T4
		l2avgl4.Run();	
		
		
		if(PrintResultsAtEveryN4){
			double l4norm1 = 1./double(i+1);
			L4AvgPP<false> l4avgpp1(l4, arg->Rmax(), l4norm1, arg->nLinksLvl1());
			Array<complexd>* res = l4avgpp1.Run();
			cout << res << endl;
			delete res;
		}
	}
	delete dev_lat;
	delete dev_mhit;
	delete l2;
	//Average tensor T4 and Calculate P(0)*conj(P(r))
	Array<complexd> *pp = l4avgpp.Run();
	delete l4;

	std::ofstream fileout;
	std::string filename = "Pot_mlevel_" + GetLatticeNameI();
	filename += "_" + ToString(arg->nLinksLvl1()) + "_" + ToString(arg->StepsLvl1());
	filename += "_" + ToString(arg->UpdatesLvl1()) + "_" + ToString(arg->nLinksLvl0());
	filename += "_" + ToString(arg->StepsLvl0()) + "_" + ToString(arg->UpdatesLvl0());
	filename += "_" + ToString(arg->nUpdatesMetropolis()) + "_" + ToString(arg->nUpdatesOvr());
	filename += "_" + ToString(arg->Rmax());
	filename += ".dat";
	fileout.open (filename.c_str());
	if (!fileout.is_open()) {
		std::cout << "Error opening file: " << filename << std::endl;
		exit(1);
	}
	cout << "Saving data to " << filename << endl;
	fileout << std::scientific;
	fileout.precision(14);
	cout << std::scientific;
	cout << std::setprecision(14);
	
	for(int r = 0; r < arg->Rmax(); ++r){
		cout << r << '\t' << pp->at(r) << endl;
		fileout << r << '\t' << pp->at(r) << endl;
	}
	
	fileout.close();
	a0.stop();
	std::cout << "time " << a0.getElapsedTime() << " s" << endl;
	
	return pp;
}























std::tuple<Array<complexd>*, Array<complexd>*> MultiLevelField(Array<double> *lat, CudaRNG *rng_state, MLArg *arg, bool PrintResultsAtEveryN4){
	Timer a0; a0.start();
	
	arg->check();	
	arg->print();
	
	
	Array<double>* dev_lat = new Array<double>(Device);
	dev_lat->Copy(lat);

	int nl2 = Grid(TDir())/arg->nLinksLvl0();
	int sl2 = nl2*(Dirs()-1)*arg->Rmax()*SpatialVolume();
	Array<complexd> *l2 = new Array<complexd>(Device, sl2);
	int nl4 = Grid(TDir())/arg->nLinksLvl1();
	size_t sl4 = nl4*(Dirs()-1)*arg->Rmax()*SpatialVolume();
	Array<complexd> *l4 = new Array<complexd>(Device, sl4);
	
	
	// metropolis and overrelaxation algorithm
	Metropolis_ML mtp(dev_lat, rng_state);
	OverRelaxation_ML ovr(dev_lat);
		
	const bool multihit = true;
	Polyakov_Volume<multihit> mhitVol(dev_lat);
	Array<complexd>* dev_mhit;
	
	double l2norm = 1./double(arg->StepsLvl0());
	L2AvgL4ML l2avgl4(l2, l4, sl4, arg->Rmax(), l2norm, arg->nLinksLvl0(), arg->nLinksLvl1());
	double l4norm = 1./double(arg->StepsLvl1());
	L4AvgPP<true> l4avgpp(l4, arg->Rmax(), l4norm, arg->nLinksLvl1());

	l4->Clear();
	for(int i = 0; i < arg->StepsLvl1(); ++i){
		cout << "Iter of l4: " << i << endl;
		//Update the lattice k4 times freezing spacial links in layers with t multiple of 4
		for(int j = 0; j < arg->UpdatesLvl1(); ++j){
			mtp.Run(arg->nUpdatesMetropolis(), arg->nLinksLvl1());
			ovr.Run(arg->nUpdatesOvr(), arg->nLinksLvl1());
		}
		l2->Clear();
		for(int k = 0; k < arg->StepsLvl0(); ++k){		
			//Update the lattice k2 times freezing spacial links in layers with t multiple of 2
			for(int l = 0; l < arg->UpdatesLvl0(); ++l){
				mtp.Run(arg->nUpdatesMetropolis(), arg->nLinksLvl0());
				ovr.Run(arg->nUpdatesOvr(), arg->nLinksLvl0());
			}
			//Extract temporal links and apply MultiHit
			dev_mhit = mhitVol.Run();			
			//Calculate tensor T2
			L2ML l2ml(dev_mhit, l2, sl2, arg->Rmax(), arg->nLinksLvl0());
			//L2ML1<multihit> l2ml(dev_lat, l2, sl2, arg->Rmax(), arg->nLinksLvl0());  // <--- SLOW
			l2ml.Run();
		}
		//Average tensor T2 and Calculate tensor T4
		l2avgl4.Run();	
		
		
		if(PrintResultsAtEveryN4){
			double l4norm1 = 1./double(i+1);
			L4AvgPP<false> l4avgpp1(l4, arg->Rmax(), l4norm1, arg->nLinksLvl1());
			Array<complexd>* res = l4avgpp1.Run();
			cout << res << endl;
			delete res;
		}
	}
	delete dev_lat;
	delete dev_mhit;
	delete l2;
	//Average tensor T4 and Calculate P(0)*conj(P(r))
	Array<complexd> *pp = l4avgpp.Run();
	delete l4;
	Array<complexd> *ppfield = l4avgpp.getField();
	auto data = std::make_tuple(pp, ppfield);

	std::ofstream fileout;
	std::string filename = "Pot_mlevel_" + GetLatticeNameI();
	filename += "_" + ToString(arg->nLinksLvl1()) + "_" + ToString(arg->StepsLvl1());
	filename += "_" + ToString(arg->UpdatesLvl1()) + "_" + ToString(arg->nLinksLvl0());
	filename += "_" + ToString(arg->StepsLvl0()) + "_" + ToString(arg->UpdatesLvl0());
	filename += "_" + ToString(arg->nUpdatesMetropolis()) + "_" + ToString(arg->nUpdatesOvr());
	filename += "_" + ToString(arg->Rmax());
	filename += ".dat";
	fileout.open (filename.c_str());
	if (!fileout.is_open()) {
		std::cout << "Error opening file: " << filename << std::endl;
		exit(1);
	}
	cout << "Saving data to " << filename << endl;
	fileout << std::scientific;
	fileout.precision(14);
	cout << std::scientific;
	cout << std::setprecision(14);
	
	for(int r = 0; r < arg->Rmax(); ++r){
		cout << r << '\t' << pp->at(r) << endl;
		fileout << r << '\t' << pp->at(r) << endl;
	}
	
	fileout.close();
	a0.stop();
	std::cout << "time " << a0.getElapsedTime() << " s" << endl;
	
	return data;
}




}
