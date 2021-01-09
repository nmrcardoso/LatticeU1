
#include <cuda.h>
#include <curand_kernel.h>
#include "cuda_error_check.h"

#include "random.h"
#include "parameters.h"


namespace U1{

std::random_device device;
std::mt19937 *generator;


__global__ void 
kernel_random(cuRNGState *state, int seed){
    size_t id = threadIdx.x + blockDim.x * blockIdx.x;
    if( id >= HalfVolume() ) return ;
    curand_init(seed, id, 0, &state[id]);
}


cuRNGState* Init_Device_RNG(int seed){
	cuRNGState *rng_state = (cuRNGState*)dev_malloc(HalfVolume()*sizeof(cuRNGState));
	// kernel number of threads per block and number os blocks
	int threads = 128;
	int blocks = (HalfVolume() + threads - 1) / threads;	
	//Initialize cuda rng	
	kernel_random<<<blocks,threads>>>(rng_state, 1234);
	return rng_state;
}

}
