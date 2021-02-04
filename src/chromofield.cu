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


#include "tune.h"

#include "array.h"


using namespace std;


namespace U1{







struct ChromoFieldArg{
  complexd *wloop;
  complexd *field;
  complexd *pl;
  complexd *plaq;
  int volume;
  int Rmin;
  int Rmax;
  int Tmax;
  int nx;
  int ny;
};





__global__ void kernel_ChromoField(ChromoFieldArg arg){

	size_t id = threadIdx.x + blockDim.x * blockIdx.x;
		
	int fieldoffset = arg.nx * arg.ny; 
		  
for(int radius = arg.Rmin; radius < arg.Rmax; radius++){
for(int t = 0; t < arg.Tmax; t++){

  int radiusoffset = 6 * fieldoffset * radius +  6 * fieldoffset * arg.Rmax * t;
  
 // Real value = 0.0;
  for(int dirz = 0; dirz < Dirs()-1; dirz++){
    complexd loop = 0.0;
    if(id < arg.volume){  
      loop = arg.wloop[id + arg.volume * dirz + arg.volume * (Dirs()-1) * (radius + arg.Rmax * t)];
    }
    
    int x[4];
    indexNO(id, x);
    x[dirz] = (x[dirz] + (radius+1) / 2) % Grid(dirz);
    
	int EvenRadius = (radius+1)%2;
    for( int ix = 0; ix < arg.nx; ++ix )
    for( int iy = 0; iy < arg.ny; ++iy ) {
    
      complexd field[6];
      for(int dd = 0; dd < 6; dd++) field[dd] = 0.0;

      	if(EvenRadius){
		  for(int dirx = 0; dirx < Dirs()-1; dirx++){
		    if(dirx==dirz) continue;
		    int diry = 0;
		    for(int diryy = 0; diryy < Dirs()-1; diryy++) if(dirx != diryy && dirz != diryy) diry = diryy;
		  
	  		  int pos = indexId(x);
	  		  int s = indexNO_neg(pos, dirx, ix - arg.nx / 2, dirz, iy - arg.ny / 2);
		    
		    if(id < arg.volume){
		      //Ex^2
		      complexd plaq = arg.plaq[s + dirx * arg.volume];
		      field[0] += plaq * 0.5;
		      //Ey^2
		      int s1 = indexNO_neg(s, diry, -1);
		      plaq = arg.plaq[s + diry * arg.volume];
		      plaq += arg.plaq[s1 + diry * arg.volume];
		      field[1] += plaq * 0.5;
		      //Ez^2
		      plaq = arg.plaq[s + dirz * arg.volume];
		      field[2] += plaq * 0.5;
		      //Bx^2
		      plaq = arg.plaq[s + (3 + dirx) * arg.volume];
		      plaq += arg.plaq[s1 + (3 + dirx) * arg.volume];
		      field[3] += plaq * 0.25;
		      //By^2
		      plaq = arg.plaq[s + (3 + diry) * arg.volume];
		      field[4] += plaq * 0.25;
		      //Bz^2
		      plaq = arg.plaq[s + (3 + dirz) * arg.volume];
		      plaq += arg.plaq[s1 + (3 + dirz) * arg.volume];
		      field[5] += plaq * 0.25;
		    }
		  }
		  for(int dd = 0; dd < 6; dd++){
		  	// TOCHECK!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!
		  	field[dd].real() *= loop.real();   //Ex^2
		  	field[dd].imag() *= loop.real();   //Ex
		  }
		   
		  complexd aggregate[6];
		  for(int dd = 0; dd < 6; dd++){
		    reduce_local_block_1d<complexd>(aggregate[dd], field[dd]);
		  }
		  
		  if (threadIdx.x == 0){
		  //accum Ex^2
		    int id0 = ix + iy * arg.nx;
		    CudaAtomicAdd(arg.field + id0 + radiusoffset, aggregate[0]);
		    int id1 =  ((ix + 1) % arg.nx) + iy * arg.nx;
		    CudaAtomicAdd(arg.field + id1 + radiusoffset, aggregate[0]);
		  //accum Ey^2
		    CudaAtomicAdd(arg.field + id0 + 1 * fieldoffset + radiusoffset, aggregate[1]);
		  //accum Ez^2		    
		    CudaAtomicAdd(arg.field + id0 + 2 * fieldoffset + radiusoffset, aggregate[2]);
		    id1 =  ix + ((iy + 1) % arg.ny) * arg.nx;
		    CudaAtomicAdd(arg.field + id1 + 2 * fieldoffset + radiusoffset, aggregate[2]);    
		  //accum Bx^2
		    CudaAtomicAdd(arg.field + id0 + 3 * fieldoffset + radiusoffset, aggregate[3]);
		    id1 =  ix + ((iy + 1) % arg.ny) * arg.nx;
		    CudaAtomicAdd(arg.field + id1 + 3 * fieldoffset + radiusoffset, aggregate[3]); 
		  //accum By^2
		    CudaAtomicAdd(arg.field + id0 + 4 * fieldoffset + radiusoffset, aggregate[4]);
		    id1 =  ((ix + 1) % arg.nx) + iy * arg.nx;
		    CudaAtomicAdd(arg.field + id1 + 4 * fieldoffset + radiusoffset, aggregate[4]);
		    id1 =  ix + ((iy + 1) % arg.ny) * arg.nx;
		    CudaAtomicAdd(arg.field + id1 + 4 * fieldoffset + radiusoffset, aggregate[4]);
		    id1 =  ((ix + 1) % arg.nx) + ((iy + 1) % arg.ny) * arg.nx;
		    CudaAtomicAdd(arg.field + id1 + 4 * fieldoffset + radiusoffset, aggregate[4]);		    
		  //accum Bz^2
		    CudaAtomicAdd(arg.field + id0 + 5 * fieldoffset + radiusoffset, aggregate[5]);
		    id1 =  ((ix + 1) % arg.nx) + iy * arg.nx;
		    CudaAtomicAdd(arg.field + id1 + 5 * fieldoffset + radiusoffset, aggregate[5]);
		}
	  }
	  else{
		  for(int dirx = 0; dirx < Dirs()-1; dirx++){
			if(dirx==dirz) continue;
			int diry = 0;
			for(int diryy = 0; diryy < Dirs()-1; diryy++) if(dirx != diryy && dirz != diryy) diry = diryy;

			int pos = indexId(x);
			int s = indexNO_neg(pos, dirx, ix - arg.nx / 2, dirz, iy - arg.ny / 2);

		    if(id < arg.volume){
		      //Ex^2
		      complexd plaq = arg.plaq[s + dirx * arg.volume];
		      field[0] += plaq * 0.25;
		      //Ey^2
		      plaq = arg.plaq[s + diry * arg.volume];
		      int s1 = indexNO_neg(s, diry, -1);
		      plaq += arg.plaq[s1 + diry * arg.volume];
		      field[1] += plaq * 0.25;
		      //Ez^2
		      plaq = arg.plaq[s + dirz * arg.volume];
		      field[2] += plaq;
		      //Bx^2
		      plaq = arg.plaq[s + (3 + dirx) * arg.volume];
		      plaq += arg.plaq[s1 + (3 + dirx) * arg.volume];
		      field[3] += plaq * 0.5;
		      //By^2
		      plaq = arg.plaq[s + (3 + diry) * arg.volume];
		      field[4] += plaq * 0.5;
		      //Bz^2	   
		      plaq = arg.plaq[s + (3 + dirz) * arg.volume];
		      plaq += arg.plaq[s1 + (3 + dirz) * arg.volume];
		      field[5] += plaq * 0.125; 
		    }
		  }
		  
		  for(int dd = 0; dd < 6; dd++){
		  	// TOCHECK!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!
		  	field[dd].real() *= loop.real();   //Ex^2
		  	field[dd].imag() *= loop.real();   //Ex	  
		  }
		  complexd aggregate[6];
		  for(int dd = 0; dd < 6; dd++){
		    reduce_local_block_1d<complexd>(aggregate[dd], field[dd]);
		  }
		    
		  	     
		  
		  if (threadIdx.x == 0){
		  //accum Ex^2
		    int id0 = ix + iy * arg.nx;
		    CudaAtomicAdd(arg.field + id0 + radiusoffset, aggregate[0]);
		    int id1 =  ((ix + 1) % arg.nx) + iy * arg.nx;
		    CudaAtomicAdd(arg.field + id1 + radiusoffset, aggregate[0]); 
			id1 =  ix + ((iy - 1 + arg.ny) % arg.ny) * arg.nx;
		    CudaAtomicAdd(arg.field + id1 + radiusoffset, aggregate[0]);
			id1 =  ((ix + 1) % arg.nx) + ((iy - 1 + arg.ny) % arg.ny) * arg.nx;
		    CudaAtomicAdd(arg.field + id1 + radiusoffset, aggregate[0]);
		  //accum Ey^2
		    CudaAtomicAdd(arg.field + id0 + 1 * fieldoffset + radiusoffset, aggregate[1]);
			id1 =  ix + ((iy - 1 + arg.ny) % arg.ny) * arg.nx;
		    CudaAtomicAdd(arg.field + id1 + 1 * fieldoffset + radiusoffset, aggregate[1]);
		  //accum Ez^2
		    CudaAtomicAdd(arg.field + id0 + 2 * fieldoffset + radiusoffset, aggregate[2]);
		  //accum Bx^2
		    CudaAtomicAdd(arg.field + id0 + 3 * fieldoffset + radiusoffset, aggregate[3]);
		  //accum By^2
		    CudaAtomicAdd(arg.field + id0 + 4 * fieldoffset + radiusoffset, aggregate[4]);
		    id1 =  ((ix + 1) % arg.nx) + iy * arg.nx;
		    CudaAtomicAdd(arg.field + id1 + 4 * fieldoffset + radiusoffset, aggregate[4]);
		  //accum Bz^2
		    CudaAtomicAdd(arg.field + id0 + 5 * fieldoffset + radiusoffset, aggregate[5]);
		    id1 =  ((ix + 1) % arg.nx) + iy * arg.nx;
		    CudaAtomicAdd(arg.field + id1 + 5 * fieldoffset + radiusoffset, aggregate[5]);
			id1 =  ix + ((iy - 1 + arg.ny) % arg.ny) * arg.nx;
		    CudaAtomicAdd(arg.field + id1 + 5 * fieldoffset + radiusoffset, aggregate[5]);
			id1 =  ((ix + 1) % arg.nx) + ((iy - 1 + arg.ny) % arg.ny) * arg.nx;
		    CudaAtomicAdd(arg.field + id1 + 5 * fieldoffset + radiusoffset, aggregate[5]);
		  }
	  }
      __syncthreads();
    }
    //value += loop;
  }
/*	Real pl = 0.;
	pl = BlockReduce(temp_storage).Reduce(value, Summ<Real>());
	__syncthreads();
	if (threadIdx.x == 0) CudaAtomicAdd(arg.pl, pl);*/
}
}
}


__global__ void kernel_ChromoFieldMidFluxTube(ChromoFieldArg arg){

	size_t id = threadIdx.x + blockDim.x * blockIdx.x;
		
	int fieldoffset = arg.nx * arg.ny; 
		  
for(int radius = arg.Rmin; radius < arg.Rmax; radius++){
for(int t = 0; t < arg.Tmax; t++){

  int radiusoffset = 6 * fieldoffset * radius +  6 * fieldoffset * arg.Rmax * t;
  
 // Real value = 0.0;
  for(int dirz = 0; dirz < Dirs()-1; dirz++){
    complexd loop = 0.0;
    if(id < arg.volume){  
      loop = arg.wloop[id + arg.volume * dirz + arg.volume * (Dirs()-1) * (radius + arg.Rmax * t)];
    }
    
    int x[4];
    indexNO(id, x);
    x[dirz] = (x[dirz] + (radius+1) / 2) % Grid(dirz);
    
	int EvenRadius = (radius+1)%2;
    for( int ix = 0; ix < arg.nx; ++ix )
    for( int iy = 0; iy < arg.ny; ++iy ) {
    
      complexd field[6];
      for(int dd = 0; dd < 6; dd++) field[dd] = 0.0;

      	if(EvenRadius){
		  for(int dirx = 0; dirx < Dirs()-1; dirx++){
		    if(dirx==dirz) continue;
		    int diry = 0;
		    for(int diryy = 0; diryy < Dirs()-1; diryy++) if(dirx != diryy && dirz != diryy) diry = diryy;
		  
	  		  int pos = indexId(x);
	  		  int s = indexNO_neg(pos, dirx, ix - arg.nx / 2, diry, iy - arg.ny / 2);
		    	    
		    
		    if(id < arg.volume){
		      //Ex^2
		      complexd plaq = arg.plaq[s + dirx * arg.volume];
		      field[0] += plaq * 0.5;
		      //Ey^2
		      plaq = arg.plaq[s + diry * arg.volume];
		      field[1] += plaq * 0.5;
		      //Ez^2
		      int s1 = indexNO_neg(s, dirz, -1);
		      plaq = arg.plaq[s + dirz * arg.volume];
		      plaq += arg.plaq[s1 + dirz * arg.volume];
		      field[2] += plaq * 0.5;
		      //Bx^2
		      plaq = arg.plaq[s + (3 + dirx) * arg.volume];
		      plaq += arg.plaq[s1 + (3 + dirx) * arg.volume];
		      field[3] += plaq * 0.25;
		      //By^2
		      plaq = arg.plaq[s + (3 + diry) * arg.volume];
		      plaq += arg.plaq[s1 + (3 + diry) * arg.volume];
		      field[4] += plaq * 0.25;
		      //Bz^2
		      plaq = arg.plaq[s + (3 + dirz) * arg.volume];
		      field[5] += plaq * 0.25;
		    }
		    
		  } 
		  for(int dd = 0; dd < 6; dd++){
		  	field[dd].real() *= loop.real();   //Ex^2
		  	field[dd].imag() *= loop.real();   //Ex
		  }
		   
		  complexd aggregate[6];
		  for(int dd = 0; dd < 6; dd++){
		    reduce_local_block_1d<complexd>(aggregate[dd], field[dd]);
		  }
		  
		  if (threadIdx.x == 0){
		  //accum Ex^2
		    int id0 = ix + iy * arg.nx;
		    CudaAtomicAdd(arg.field + id0 + radiusoffset, aggregate[0]);
		    int id1 =  ((ix + 1) % arg.nx) + iy * arg.nx;
		    CudaAtomicAdd(arg.field + id1 + radiusoffset, aggregate[0]);
		  //accum Ey^2
		    CudaAtomicAdd(arg.field + id0 + 1 * fieldoffset + radiusoffset, aggregate[1]);
		    id1 =  ix + ((iy + 1) % arg.ny) * arg.nx;
		    CudaAtomicAdd(arg.field + id1 + 1 * fieldoffset + radiusoffset, aggregate[1]); 
		  //accum Ez^2
		    CudaAtomicAdd(arg.field + id0 + 2 * fieldoffset + radiusoffset, aggregate[2]);
		  //accum Bx^2
		    CudaAtomicAdd(arg.field + id0 + 3 * fieldoffset + radiusoffset, aggregate[3]);
		    id1 =  ix + ((iy + 1) % arg.ny) * arg.nx;
		    CudaAtomicAdd(arg.field + id1 + 3 * fieldoffset + radiusoffset, aggregate[3]); 
		  //accum By^2
		    CudaAtomicAdd(arg.field + id0 + 4 * fieldoffset + radiusoffset, aggregate[4]);
		    id1 =  ((ix + 1) % arg.nx) + iy * arg.nx;
		    CudaAtomicAdd(arg.field + id1 + 4 * fieldoffset + radiusoffset, aggregate[4]);
		  //accum Bz^2
		    CudaAtomicAdd(arg.field + id0 + 5 * fieldoffset + radiusoffset, aggregate[5]);
		    id1 =  ((ix + 1) % arg.nx) + iy * arg.nx;
		    CudaAtomicAdd(arg.field + id1 + 5 * fieldoffset + radiusoffset, aggregate[5]);
		    id1 =  ix + ((iy + 1) % arg.ny) * arg.nx;
		    CudaAtomicAdd(arg.field + id1 + 5 * fieldoffset + radiusoffset, aggregate[5]);
		    id1 =  ((ix + 1) % arg.nx) + ((iy + 1) % arg.ny) * arg.nx;
		    CudaAtomicAdd(arg.field + id1 + 5 * fieldoffset + radiusoffset, aggregate[5]);
		  } 
	  }
	  else{
		  for(int dirx = 0; dirx < Dirs()-1; dirx++){
			if(dirx==dirz) continue;
			int diry = 0;
			for(int diryy = 0; diryy < Dirs()-1; diryy++) if(dirx != diryy && dirz != diryy) diry = diryy;

			int pos = indexId(x);
			int s = indexNO_neg(pos, dirx, ix - arg.nx / 2, diry, iy - arg.ny / 2);

		    if(id < arg.volume){
		      //Ex^2
		      complexd plaq = arg.plaq[s + dirx * arg.volume];
		      int s1 = indexNO_neg(s, dirz, 1);
		      plaq += arg.plaq[s1 + dirx * arg.volume];
		      field[0] += plaq * 0.25;
		      //Ey^2
		      plaq = arg.plaq[s + diry * arg.volume];
		      plaq += arg.plaq[s1 + diry * arg.volume];
		      field[1] += plaq * 0.25;
		      //Ez^2
		      plaq = arg.plaq[s + dirz * arg.volume];
		      field[2] += plaq;
		      //Bx^2
		      plaq = arg.plaq[s + (3 + dirx) * arg.volume];
		      field[3] += plaq * 0.5;
		      //By^2
		      plaq = arg.plaq[s + (3 + diry) * arg.volume];
		      field[4] += plaq * 0.5;
		      //Bz^2
		      plaq = arg.plaq[s + (3 + dirz) * arg.volume];
		      plaq += arg.plaq[s1 + (3 + dirz) * arg.volume];
		      field[5] += plaq * 0.125;	    
		    }
		  }
		    
		  for(int dd = 0; dd < 6; dd++){
		  	field[dd].real() *= loop.real();   //Ex^2
		  	field[dd].imag() *= loop.real();   //Ex		  
		  }
		  complexd aggregate[6];
		  for(int dd = 0; dd < 6; dd++){
		    reduce_local_block_1d<complexd>(aggregate[dd], field[dd]);
		  }
		  
		  if (threadIdx.x == 0){
		  //accum Ex^2
		    int id0 = ix + iy * arg.nx;
		    CudaAtomicAdd(arg.field + id0 + radiusoffset, aggregate[0]);
		    int id1 =  ((ix + 1) % arg.nx) + iy * arg.nx;
		    CudaAtomicAdd(arg.field + id1 + radiusoffset, aggregate[0]); 
		  //accum Ey^2
		    CudaAtomicAdd(arg.field + id0 + 1 * fieldoffset + radiusoffset, aggregate[1]);
			id1 =  ix + ((iy + 1) % arg.ny) * arg.nx;
		    CudaAtomicAdd(arg.field + id1 + 1 * fieldoffset + radiusoffset, aggregate[1]);
		  //accum Ez^2
		    CudaAtomicAdd(arg.field + id0 + 2 * fieldoffset + radiusoffset, aggregate[2]);
		  //accum Bx^2
		    CudaAtomicAdd(arg.field + id0 + 3 * fieldoffset + radiusoffset, aggregate[3]);
		    id1 =  ix + ((iy + 1) % arg.ny) * arg.nx;
		    CudaAtomicAdd(arg.field + id1 + 3 * fieldoffset + radiusoffset, aggregate[3]); 
		  //accum By^2
		    CudaAtomicAdd(arg.field + id0 + 4 * fieldoffset + radiusoffset, aggregate[4]);
		    id1 =  ((ix + 1) % arg.nx) + iy * arg.nx;
		    CudaAtomicAdd(arg.field + id1 + 4 * fieldoffset + radiusoffset, aggregate[4]);
		  //accum Bz^2
		    CudaAtomicAdd(arg.field + id0 + 5 * fieldoffset + radiusoffset, aggregate[5]);
		    id1 =  ((ix + 1) % arg.nx) + iy * arg.nx;
		    CudaAtomicAdd(arg.field + id1 + 5 * fieldoffset + radiusoffset, aggregate[5]); 
		    id1 =  ix + ((iy + 1) % arg.ny) * arg.nx;
		    CudaAtomicAdd(arg.field + id1 + 5 * fieldoffset + radiusoffset, aggregate[5]); 
		    id1 =  ((ix + 1) % arg.nx) + ((iy + 1) % arg.ny) * arg.nx;
		    CudaAtomicAdd(arg.field + id1 + 5 * fieldoffset + radiusoffset, aggregate[5]); 
		  } 
	  }
      __syncthreads();
    }
  }
}
}
}





	

template <bool chargeplane>
class ChromoField: Tunable{
private:
   ChromoFieldArg arg;
   Array<complexd> *chromofield;
   Array<complexd> *field;
   int size;
   double timesec;
#ifdef TIMMINGS
    Timer ChromoFieldtime;
#endif

   unsigned int sharedBytesPerThread() const { return sizeof(complexd); }
   unsigned int sharedBytesPerBlock(const TuneParam &param) const { return 0; }
   bool tuneSharedBytes() const { return false; } // Don't tune shared memory
   bool tuneGridDim() const { return false; } // Don't tune the grid dimensions.
   unsigned int minThreads() const { return size; }
   void apply(const cudaStream_t &stream){
	TuneParam tp = tuneLaunch(*this, getTuning(), getVerbosity());
	field->Clear();
	if(chargeplane) kernel_ChromoField<<<tp.grid, tp.block, tp.shared_bytes, stream>>>(arg);
	else kernel_ChromoFieldMidFluxTube<<<tp.grid, tp.block, tp.shared_bytes, stream>>>(arg);
}

public:
   ChromoField(Array<complexd> *wloop, Array<complexd> *plaqfield, Array<complexd> *chromofield, int Rmin, int Rmax, int Tmax, int nx, int ny, int volume): chromofield(chromofield){
	size = volume;
	timesec = 0.0;
	arg.nx = nx;
	arg.ny = ny;
	arg.volume = volume;
	arg.Rmax = Rmax;
	arg.Rmin = Rmin;
	arg.Tmax = Tmax;
	arg.wloop = wloop->getPtr();
	arg.plaq = plaqfield->getPtr();
	field = new Array<complexd>(Device, chromofield->Size());
	arg.field = field->getPtr();
	cout << arg.wloop << '\t' << arg.plaq << '\t' << arg.field << endl;
	cout << arg.nx << '\t' << arg.ny << '\t' << arg.Rmax << '\t' << arg.Tmax << endl; 
}
   ~ChromoField(){delete field;}
   void Run(const cudaStream_t &stream){
#ifdef TIMMINGS
    ChromoFieldtime.start();
#endif
    apply(stream);
    cudaDevSync();
    cudaCheckError("Kernel execution failed");
    chromofield->Copy(field);
    //normalize!!!!!!
	int plane = arg.nx * arg.ny;
	int fsize = 6 * plane;
	for(int t = 0; t < arg.Tmax; t++)
	for(int r = arg.Rmin; r < arg.Rmax; r++){
		int id = r + arg.Rmax * t;
		for(int f = 0; f < fsize; f++)
		  chromofield->at(f + id * fsize) /= double(6 * size);
	}
#ifdef TIMMINGS
	cudaDevSync( );
    ChromoFieldtime.stop();
    timesec = ChromoFieldtime.getElapsedTimeInSec();
#endif
}

   void Run(){ Run(0);}
   double flops(){	return ((double)flop() * 1.0e-9) / timesec;}
   double bandwidth(){	return (double)bytes() / (timesec * (double)(1 << 30));}
   long long flop() const { return 0;}
   long long bytes() const{ return 0;}
   double get_time(){	return timesec;}
   void stat(){	cout << "ChromoField:  " <<  get_time() << " s\t"  << bandwidth() << " GB/s\t" << flops() << " GFlops"  << endl;}
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
  void postTune() {  }

};






template<bool chargeplane>
void CalcChromoField(Array<complexd> *wloop, Array<complexd> *plaqfield, Array<complexd> *field, int Rmin, int Rmax, int Tmax, int nx, int ny, int volume){
	Timer mtime;
	mtime.start(); 
	ChromoField<chargeplane> cfield(wloop, plaqfield, field, Rmin, Rmax, Tmax, nx, ny, volume);
	cfield.Run();
	cudaDevSync( );
	mtime.stop();
	cout << "Time ChromoField:  " <<  mtime.getElapsedTimeInSec() << " s"  << endl;
}




void CalcChromoFieldWL(Array<complexd> *wloop, Array<complexd> *plaqfield, Array<complexd> *field, int Rmin, int Rmax, int Tmax, int nx, int ny, bool chargeplane){
	if(Dirs() < 4){
		cout << "Only implemented for 4D lattice..." << endl;
		Finalize(1);
	}
	if(chargeplane) CalcChromoField<true>(wloop, plaqfield, field, Rmin, Rmax, Tmax, nx, ny, Volume());
	else CalcChromoField<false>(wloop, plaqfield, field, Rmin, Rmax, Tmax, nx, ny, Volume());
}



void CalcChromoFieldPP(Array<complexd> *ploop, Array<complexd> *plaqfield, Array<complexd> *field, int Rmin, int Rmax, int nx, int ny, bool chargeplane){
	if(Dirs() < 4){
		cout << "Only implemented for 4D lattice..." << endl;
		Finalize(1);
	}
	if(chargeplane) CalcChromoField<true>(ploop, plaqfield, field, Rmin, Rmax, 1, nx, ny, SpatialVolume());
	else CalcChromoField<false>(ploop, plaqfield, field, Rmin, Rmax, 1, nx, ny, SpatialVolume());
}



}
