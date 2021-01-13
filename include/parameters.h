#ifndef __PARAMETERS__
#define __PARAMETERS__

#include <fstream>
#include <string>
#include <sstream>

#include <cuda.h>
#include <curand_kernel.h>
#include "cuda_error_check.h"


namespace U1{



#if defined(XORWOW)
typedef struct curandStateXORWOW cuRNGState;
#elif defined(MRG32k3a)
typedef struct curandStateMRG32k3a cuRNGState;
#else
typedef struct curandStateMRG32k3a cuRNGState;
#endif


#define  InlineHostDevice inline  __host__   __device__
#define ConstDeviceMem __constant__




TuneMode getTuning();
Verbosity getVerbosity();

void setTuning(TuneMode kerneltunein);
void setVerbosity(Verbosity verbosein);

dim3 GetBlockDim(size_t threads, size_t size);





std::string GetLatticeName();

std::string GetLatticeNameI();




namespace DEVPARAMS{
	extern ConstDeviceMem   double   Beta;
	extern ConstDeviceMem   double   Aniso;
	extern ConstDeviceMem   int DIRS;
	extern ConstDeviceMem   int TDir;
	extern ConstDeviceMem   int volume;
	extern ConstDeviceMem   int half_volume;
	extern ConstDeviceMem   int spatial_volume;
	extern ConstDeviceMem   int Grid[4];
}

namespace PARAMS{
	extern double Beta;
	extern double Aniso;
	extern int DIRS;
	extern int TDir;
	extern int volume;
	extern int half_volume;
	extern int spatial_volume;
	extern int Grid[4];
	extern int iter;
	extern double accept_ratio;
	extern int ovrn;
	extern int metrop;
    extern cudaDeviceProp deviceProp;
}


InlineHostDevice int Volume(){
    #ifdef __CUDA_ARCH__
    return DEVPARAMS::volume;
    #else
    return PARAMS::volume;
    #endif
}
InlineHostDevice int HalfVolume(){
    #ifdef __CUDA_ARCH__
    return DEVPARAMS::half_volume;
    #else
    return PARAMS::half_volume;
    #endif
}
InlineHostDevice int SpatialVolume(){
    #ifdef __CUDA_ARCH__
    return DEVPARAMS::spatial_volume;
    #else
    return PARAMS::spatial_volume;
    #endif
}
InlineHostDevice int Grid(int dim){
    #ifdef __CUDA_ARCH__
    return DEVPARAMS::Grid[dim];
    #else
    return PARAMS::Grid[dim];
    #endif
}
InlineHostDevice double Beta(){
    #ifdef __CUDA_ARCH__
    return DEVPARAMS::Beta;
    #else
    return PARAMS::Beta;
    #endif
}
InlineHostDevice int Dirs(){
    #ifdef __CUDA_ARCH__
    return DEVPARAMS::DIRS;
    #else
    return PARAMS::DIRS;
    #endif
}
InlineHostDevice int TDir(){
    #ifdef __CUDA_ARCH__
    return DEVPARAMS::TDir;
    #else
    return PARAMS::TDir;
    #endif
}
InlineHostDevice double Aniso(){
    #ifdef __CUDA_ARCH__
    return DEVPARAMS::Aniso;
    #else
    return PARAMS::Aniso;
    #endif
}


	
	void SetupGPU_Parameters();

template<class T>
inline std::string ToString(const T number){
    std::stringstream ss;//create a stringstream
    ss << number;//add number to the stream
    return ss.str();//return a string with the contents of the stream
}
template<>
inline std::string ToString<double>(const double number){
    std::stringstream ss;//create a stringstream
	//ss.precision(2);
    ss << number;//add number to the stream
    return ss.str();//return a string with the contents of the stream
}
template<>
inline std::string ToString<float>(const float number){
    std::stringstream ss;//create a stringstream
	//ss.precision(2);
    ss << number;//add number to the stream
    return ss.str();//return a string with the contents of the stream
}




}
#endif
