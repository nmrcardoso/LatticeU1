#ifndef __RANDOM__
#define __RANDOM__

#include <random>
#include <cuda.h>
#include <curand_kernel.h>
#include "cuda_error_check.h"

#if defined(XORWOW)
typedef struct curandStateXORWOW cuRNGState;
#elif defined(MRG32k3a)
typedef struct curandStateMRG32k3a cuRNGState;
#else
typedef struct curandStateMRG32k3a cuRNGState;
#endif

namespace U1{

extern std::random_device device;
extern std::mt19937 *generator;




cuRNGState* Init_Device_RNG(int seed, size_t size);





class CudaRNG{
	private:
	cuRNGState *ptr;
	cuRNGState *ptr_;
	size_t size;
	int seed;
	void Release(){
		if(ptr) dev_free(ptr);
		if(ptr_) dev_free(ptr_);
	}
	public:
	CudaRNG(){ ptr = 0; size = 0; ptr_ = 0;}
	CudaRNG(int seed, size_t size):seed(seed), size(size){
		ptr = Init_Device_RNG(seed, size);
		ptr_ = 0;
	}
	cuRNGState*  getPtr(){ return ptr; };
	void Backup(){
		ptr_ = (cuRNGState*)dev_malloc(size*sizeof(cuRNGState));		
		cudaSafeCall(cudaMemcpy(ptr_, ptr, size*sizeof(cuRNGState), cudaMemcpyDeviceToDevice));
	}
	void Restore(){		
		cudaSafeCall(cudaMemcpy(ptr, ptr_, size*sizeof(cuRNGState), cudaMemcpyDeviceToDevice));
		dev_free(ptr_);
		ptr_ = 0;
	}
	~CudaRNG(){ Release(); }
};












/**
   @brief Return a random number between a and b
   @param state curand rng state
   @param a lower range
   @param b upper range
   @return  random number in range a,b
*/
template<class Real>
inline  __device__ Real Random(cuRNGState &state, Real a, Real b){
    Real res;
    return res;
}
 
template<>
inline  __device__ float Random<float>(cuRNGState &state, float a, float b){
    return a + (b - a) * curand_uniform(&state);
}

template<>
inline  __device__ double Random<double>(cuRNGState &state, double a, double b){
    return a + (b - a) * curand_uniform_double(&state);
}

/**
   @brief Return a random number between 0 and 1
   @param state curand rng state
   @return  random number in range 0,1
*/
template<class Real>
inline  __device__ Real Random(cuRNGState &state){
    Real res;
    return res;
}
 
template<>
inline  __device__ float Random<float>(cuRNGState &state){
    return curand_uniform(&state);
}

template<>
inline  __device__ double Random<double>(cuRNGState &state){
    return curand_uniform_double(&state);
}










}

#endif
