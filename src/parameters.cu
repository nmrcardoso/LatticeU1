
#include <iostream>
#include <math.h> 
#include <time.h> 
#include <random>
#include <vector> 
#include <fstream>
#include <omp.h>

#include <cuda.h>
#include <curand_kernel.h>

#include "cuda_error_check.h"
#include "enum.h"
#include "parameters.h"


namespace U1{

//static Verbosity verbose = SILENT;
//static Verbosity verbose = DEBUG_VERBOSE;

static TuneMode kerneltune = TUNE_YES;
static Verbosity verbose = VERBOSE;

TuneMode getTuning(){
  return kerneltune;
}
Verbosity getVerbosity(){
  return verbose;
}


void setTuning(TuneMode kerneltunein){
  kerneltune = kerneltunein;
}
void setVerbosity(Verbosity verbosein){
  verbose = verbosein;
}




std::string GetLatticeName(){
	std::string name = "";
	for(int i = 0; i < Dirs(); i++) name += ToString(PARAMS::Grid[i]) + "_";;
	name += "beta_" +  ToString(PARAMS::Beta);
	name += "_xi_" +  ToString(PARAMS::Aniso);	
	name += "_mtr_" +  ToString(PARAMS::metrop);
	name += "_ovr_" +  ToString(PARAMS::ovrn);
	return name;
}



std::string GetLatticeNameI(){
	std::string name = GetLatticeName();
	name += "_iter_" + ToString(PARAMS::iter);
	return name;
}




#define BLOCKSDIVUP(a, b)  (((a)+(b)-1)/(b))


dim3 GetBlockDim(size_t threads, size_t size){
	uint blockx = BLOCKSDIVUP(size, threads);
	dim3 blocks(blockx,1,1);
	return blocks;
}


#define  InlineHostDevice inline  __host__   __device__
#define ConstDeviceMem __constant__

namespace DEVPARAMS{
	ConstDeviceMem   double   Beta;
	ConstDeviceMem   double   Aniso;
	ConstDeviceMem   int DIRS;
	ConstDeviceMem   int TDir;
	ConstDeviceMem   int volume;
	ConstDeviceMem   int half_volume;
	ConstDeviceMem   int spatial_volume;
	ConstDeviceMem   int Grid[4];
	ConstDeviceMem   int Offset[4];
}

namespace PARAMS{
	double Beta;
	double Aniso;
	int DIRS;
	int TDir;
	int volume;
	int half_volume;
	int spatial_volume;
	int Grid[4];
	int Offset[4];
	int iter = 0;
	double accept_ratio = 0.;
	int ovrn = 3;
	int metrop = 1;
    cudaDeviceProp deviceProp;
}

















#define memcpyToSymbol(dev, host, type)                                 \
    cudaSafeCall(cudaMemcpyToSymbol(dev,  &host,  sizeof(type), 0, cudaMemcpyHostToDevice ));
#define memcpyToArraySymbol(dev, host, type, length)                    \
    cudaSafeCall(cudaMemcpyToSymbol(dev,  host,  length * sizeof(type), 0, cudaMemcpyHostToDevice ));



void SetupGPU_Parameters(){
	memcpyToSymbol(DEVPARAMS::Beta, PARAMS::Beta, double);
	memcpyToSymbol(DEVPARAMS::volume, PARAMS::volume, int);
	memcpyToSymbol(DEVPARAMS::half_volume, PARAMS::half_volume, int);
	memcpyToSymbol(DEVPARAMS::spatial_volume, PARAMS::spatial_volume, int);
	memcpyToSymbol(DEVPARAMS::DIRS, PARAMS::DIRS, int);
	memcpyToSymbol(DEVPARAMS::TDir, PARAMS::TDir, int);
	memcpyToArraySymbol(DEVPARAMS::Grid, PARAMS::Grid, int, 4); 
	memcpyToArraySymbol(DEVPARAMS::Offset, PARAMS::Offset, int, 4); 
	memcpyToSymbol(DEVPARAMS::Aniso, PARAMS::Aniso, double); 
}





void SetupLatticeParameters(int Nx, int Ny, int Nz, int Nt, int dirs, double beta, double aniso, int imetrop, int ovrn){
	using namespace std;
	PARAMS::DIRS = dirs; //Need to update kernels to take into account less than 4 directions
	PARAMS::TDir = PARAMS::DIRS - 1;

	for(int i = 0; i < 4; ++i) { 
		PARAMS::Grid[i] = 1; 
		PARAMS::Offset[i] = 0;
	}
	PARAMS::Grid[0] = Nx;
	if(Dirs()==2) PARAMS::Grid[1] = Nt;
	else if(Dirs() > 2) PARAMS::Grid[1] = Ny;
	if(Dirs()==3) PARAMS::Grid[2] = Nt;
	else if(Dirs() > 3) PARAMS::Grid[2] = Nz;
	if(Dirs()==4) PARAMS::Grid[3] = Nt;
	
	
	for(int i = 1; i < PARAMS::DIRS; ++i){
		if( (Grid(i)%2) != 0 ){
			std::cout << "Error: Number of points should be an even number..." << std::endl;
			Finalize(1);
		}
	}
	
	PARAMS::Offset[0] = 1;
	for(int i = 1; i < PARAMS::DIRS; ++i) { 
		PARAMS::Offset[i] = PARAMS::Grid[i-1]*PARAMS::Offset[i-1];
	}
	//for(int i = 0; i < 4; ++i) std::cout << i << '\t' << PARAMS::Offset[i] << std::endl;
	PARAMS::volume = 1;
	for(int i = 0; i < 4; ++i) PARAMS::volume *= Grid(i);	
	PARAMS::half_volume = Volume() / 2;
	PARAMS::spatial_volume = 1;
	for(int i = 0; i < TDir(); ++i) PARAMS::spatial_volume *= Grid(i);
	
	
	PARAMS::Beta = beta;
	PARAMS::Aniso = aniso;
	PARAMS::metrop = imetrop;
	PARAMS::ovrn = ovrn;
	
	
	
	
	cout << "==========================================" << endl;
	cout << "Lattice properties:" << endl;
	cout << "------------------------------------------" << endl;
	cout << "    Volume: ";
	for(int i = 0; i < Dirs(); i++){ cout << Grid(i); if(i < Dirs()-1) cout << "x"; }
	cout << endl;
	cout << "    Number of directions: " << Dirs() << endl;
	cout << "    Time indice: " << TDir() << endl;
	cout << "    Beta: " << Beta() << endl;
	cout << "    Anisotropy: " << Aniso() << endl;
	cout << "    Metropolis updates: " << PARAMS::metrop << endl;
	cout << "    Overrelaxation updates: " << PARAMS::ovrn << endl;
	
	
	
	cout << "==========================================" << endl;
	cout << "Setting up Device parameters..." << endl;	
	SetupGPU_Parameters(); // Copy parameters to GPU constant memory, need to be setup before any kernel call
}






}
