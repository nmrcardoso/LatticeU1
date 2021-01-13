
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
	memcpyToSymbol(DEVPARAMS::Aniso, PARAMS::Aniso, double); 
}





}
