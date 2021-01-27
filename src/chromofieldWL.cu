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







struct ChromoFieldArgWL{
  complexd *wloop;
  complexd *field;
  complexd *pl;
  complexd *plaq;
  int Rmax;
  int Tmax;
  int nx;
  int ny;
};





__global__ void kernel_ChromoFieldWL(ChromoFieldArgWL arg){

	size_t id = threadIdx.x + blockDim.x * blockIdx.x;
		
	int fieldoffset = arg.nx * arg.ny; 
		  
for(int radius = 0; radius < arg.Rmax; radius++){
for(int t = 0; t < arg.Tmax; t++){

  int radiusoffset = 6 * fieldoffset * radius +  6 * fieldoffset * arg.Rmax * t;
  
 // Real value = 0.0;
  for(int dir2 = 0; dir2 < Dirs()-1; dir2++){
    complexd loop = 0.0;
    if(id < Volume()){  
      loop = arg.wloop[id + Volume() * dir2 + Volume() * (Dirs()-1) * (radius + arg.Rmax * t)];
    }
    
    int x[4];
    indexNO(id, x);
    x[dir2] = (x[dir2] + (radius+1) / 2) % Grid(dir2);
    
	int EvenRadius = (radius+1)%2;
    for( int ix = 0; ix < arg.nx; ++ix )
    for( int iy = 0; iy < arg.ny; ++iy ) {
    
      complexd field[6];
      for(int dd = 0; dd < 6; dd++) field[dd] = 0.0;

      	if(EvenRadius){
		  for(int dir1 = 0; dir1 < Dirs()-1; dir1++){
		    if(dir1==dir2) continue;
		    int dir3 = 0;
		    for(int dir33 = 0; dir33 < Dirs()-1; dir33++) if(dir1 != dir33 && dir2 != dir33) dir3 = dir33;
		  
	  		  int pos = ((((x[2]) * Grid(1)) + x[1] ) * Grid(0) + x[0]);
	  		  int s = indexNO_neg(pos, dir1, ix - arg.nx / 2, dir2, iy - arg.ny / 2);
		    
		    if(id < Volume()){
		      //Ex^2
		      complexd plaq = arg.plaq[s + dir1 * Volume()];
		      field[0] += plaq;
		      //Ey^2
		      plaq = arg.plaq[s + dir2 * Volume()];
		      field[1] += plaq;
		      //Ez^2
		      int s1 = indexNO_neg(s, dir3, -1);
		      plaq = arg.plaq[s + dir3 * Volume()];
		      plaq += arg.plaq[s1 + dir3 * Volume()];
		      field[2] += plaq * 0.5;
		      //Bx^2
		      plaq = arg.plaq[s + (3 + dir1) * Volume()];
		      plaq += arg.plaq[s1 + (3 + dir1) * Volume()];
		      field[3] += plaq * 0.5;
		      //By^2
		      plaq = arg.plaq[s + (3 + dir2) * Volume()];
		      plaq += arg.plaq[s1 + (3 + dir2) * Volume()];
		      field[4] += plaq * 0.5;
		      //Bz^2
		      plaq = arg.plaq[s + (3 + dir3) * Volume()];
		      field[5] += plaq;
		    }
		  }
		  for(int dd = 0; dd < 6; dd++){
		  	// TOCHECK!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!
		  	field[dd].real() *= loop.real();
		  	field[dd].imag() *= loop.real();
		  }
		   
		  complexd aggregate[6];
		  for(int dd = 0; dd < 6; dd++){
		    reduce_local_block_1d<complexd>(aggregate[dd], field[dd]);
		  }
		  
		  if (threadIdx.x == 0){
		  //accum Ex^2
		    int id0 = ix + iy * arg.nx;
		    CudaAtomicAdd(arg.field + id0, aggregate[0]);
		    int id1 =  ((ix + 1) % arg.nx) + iy * arg.nx;
		    CudaAtomicAdd(arg.field + id1 + radiusoffset, aggregate[0]);
		  //accum Ey^2
		    CudaAtomicAdd(arg.field + id0 + fieldoffset + radiusoffset, aggregate[1]);
		    id1 =  ix + ((iy + 1) % arg.ny) * arg.nx;
		    CudaAtomicAdd(arg.field + id1 + fieldoffset + radiusoffset, aggregate[1]); 
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
		  for(int dir1 = 0; dir1 < Dirs()-1; dir1++){
			if(dir1==dir2) continue;
			int dir3 = 0;
			for(int dir33 = 0; dir33 < Dirs()-1; dir33++) if(dir1 != dir33 && dir2 != dir33) dir3 = dir33;

			int pos = indexId(x);
			int s = indexNO_neg(pos, dir1, ix - arg.nx / 2, dir2, iy - arg.ny / 2);

		    if(id < Volume()){
		      //Ex^2
		      complexd plaq = arg.plaq[s + dir1 * Volume()];
		      field[0] += plaq * 0.25;
		      //Ey^2
		      plaq = arg.plaq[s + dir2 * Volume()];
		      field[1] += plaq;
		      //Ez^2
		      plaq = arg.plaq[s + dir3 * Volume()];
		      int s1 = indexNO_neg(s, dir3, -1);
		      plaq += arg.plaq[s1 + dir3 * Volume()];
		      field[2] += plaq * 0.25;
		      //Bx^2
		      plaq = arg.plaq[s + (3 + dir1) * Volume()];
		      plaq += arg.plaq[s1 + (3 + dir1) * Volume()];
		      field[3] += plaq * 0.5;
		      //By^2
		      plaq = arg.plaq[s + (3 + dir2) * Volume()];
		      plaq += arg.plaq[s1 + (3 + dir2) * Volume()];
		      field[4] += plaq * 0.125;
		      //Bz^2
		      plaq = arg.plaq[s + (3 + dir3) * Volume()];
		      field[5] += plaq * 0.5;	    
		    }
		  }
		  
		  for(int dd = 0; dd < 6; dd++){
		  	// TOCHECK!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!
		  	field[dd].real() *= loop.real();
		  	field[dd].imag() *= loop.real();		  
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
		    CudaAtomicAdd(arg.field + id0 + fieldoffset + radiusoffset, aggregate[1]);
		  //accum Ez^2
		    CudaAtomicAdd(arg.field + id0 + 2 * fieldoffset + radiusoffset, aggregate[2]);
			id1 =  ix + ((iy - 1 + arg.ny) % arg.ny) * arg.nx;
		    CudaAtomicAdd(arg.field + id1 + 2 * fieldoffset + radiusoffset, aggregate[2]);
		  //accum Bx^2
		    CudaAtomicAdd(arg.field + id0 + 3 * fieldoffset + radiusoffset, aggregate[3]);
		  //accum By^2
		    CudaAtomicAdd(arg.field + id0 + 4 * fieldoffset + radiusoffset, aggregate[4]);
		    id1 =  ((ix + 1) % arg.nx) + iy * arg.nx;
		    CudaAtomicAdd(arg.field + id1 + 4 * fieldoffset + radiusoffset, aggregate[4]);
			id1 =  ix + ((iy - 1 + arg.ny) % arg.ny) * arg.nx;
		    CudaAtomicAdd(arg.field + id1 + 4 * fieldoffset + radiusoffset, aggregate[4]);
			id1 =  ((ix + 1) % arg.nx) + ((iy - 1 + arg.ny) % arg.ny) * arg.nx;
		    CudaAtomicAdd(arg.field + id1 + 4 * fieldoffset + radiusoffset, aggregate[4]);
		  //accum Bz^2
		    CudaAtomicAdd(arg.field + id0 + 5 * fieldoffset + radiusoffset, aggregate[5]);
		    id1 =  ((ix + 1) % arg.nx) + iy * arg.nx;
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


__global__ void kernel_ChromoFieldMidFluxTubeWL(ChromoFieldArgWL arg){

	size_t id = threadIdx.x + blockDim.x * blockIdx.x;
		
	int fieldoffset = arg.nx * arg.ny; 
		  
for(int radius = 0; radius < arg.Rmax; radius++){
for(int t = 0; t < arg.Tmax; t++){

  int radiusoffset = 6 * fieldoffset * radius +  6 * fieldoffset * arg.Rmax * t;
  
 // Real value = 0.0;
  for(int dir2 = 0; dir2 < Dirs()-1; dir2++){
    complexd loop = 0.0;
    if(id < Volume()){  
      loop = arg.wloop[id + Volume() * dir2 + Volume() * (Dirs()-1) * (radius + arg.Rmax * t)];
    }
    
    int x[4];
    indexNO(id, x);
    x[dir2] = (x[dir2] + (radius+1) / 2) % Grid(dir2);
    
	int EvenRadius = (radius+1)%2;
    for( int ix = 0; ix < arg.nx; ++ix )
    for( int iy = 0; iy < arg.ny; ++iy ) {
    
      complexd field[6];
      for(int dd = 0; dd < 6; dd++) field[dd] = 0.0;

      	if(EvenRadius){
		  for(int dir1 = 0; dir1 < Dirs()-1; dir1++){
		    if(dir1==dir2) continue;
		    int dir3 = 0;
		    for(int dir33 = 0; dir33 < Dirs()-1; dir33++) if(dir1 != dir33 && dir2 != dir33) dir3 = dir33;
		  
	  		  int pos = indexId(x);
	  		  int s = indexNO_neg(pos, dir1, ix - arg.nx / 2, dir3, iy - arg.ny / 2);
		    	    
		    
		    if(id < Volume()){
		      //Ex^2
		      complexd plaq = arg.plaq[s + dir1 * Volume()];
		      field[0] += plaq;
		      //Ey^2
		      int s1 = indexNO_neg(s, dir2, -1);
		      plaq = arg.plaq[s + dir2 * Volume()];
		      plaq += arg.plaq[s1 + dir2 * Volume()];
		      field[1] += plaq * 0.5;
		      //Ez^2
		      plaq = arg.plaq[s + dir3 * Volume()];
		      field[2] += plaq;
		      //Bx^2
		      plaq = arg.plaq[s + (3 + dir1) * Volume()];
		      plaq += arg.plaq[s1 + (3 + dir1) * Volume()];
		      field[3] += plaq * 0.5;
		      //By^2
		      plaq = arg.plaq[s + (3 + dir2) * Volume()];
		      field[4] += plaq;
		      //Bz^2
		      plaq = arg.plaq[s + (3 + dir3) * Volume()];
		      plaq += arg.plaq[s1 + (3 + dir3) * Volume()];
		      field[5] += plaq * 0.5;
		    }
		    
		  } 
		  for(int dd = 0; dd < 6; dd++){
		  	field[dd].real() *= loop.real();
		  	field[dd].imag() *= loop.real();
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
		    CudaAtomicAdd(arg.field + id0 + fieldoffset + radiusoffset, aggregate[1]);
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
		  for(int dir1 = 0; dir1 < Dirs()-1; dir1++){
			if(dir1==dir2) continue;
			int dir3 = 0;
			for(int dir33 = 0; dir33 < Dirs()-1; dir33++) if(dir1 != dir33 && dir2 != dir33) dir3 = dir33;

			int pos = indexId(x);
			int s = indexNO_neg(pos, dir1, ix - arg.nx / 2, dir3, iy - arg.ny / 2);

		    if(id < Volume()){
		      //Ex^2
		      complexd plaq = arg.plaq[s + dir1 * Volume()];
		      int s1 = indexNO_neg(s, dir2, 1);
		      plaq += arg.plaq[s1 + dir1 * Volume()];
		      field[0] += plaq * 0.25;
		      //Ey^2
		      plaq = arg.plaq[s + dir2 * Volume()];
		      field[1] += plaq;
		      //Ez^2
		      plaq = arg.plaq[s + dir3 * Volume()];
		      plaq += arg.plaq[s1 + dir3 * Volume()];
		      field[2] += plaq * 0.25;
		      //Bx^2
		      plaq = arg.plaq[s + (3 + dir1) * Volume()];
		      field[3] += plaq * 0.5;
		      //By^2
		      plaq = arg.plaq[s + (3 + dir2) * Volume()];
		      plaq += arg.plaq[s1 + (3 + dir2) * Volume()];
		      field[4] += plaq * 0.125;
		      //Bz^2
		      plaq = arg.plaq[s + (3 + dir3) * Volume()];
		      field[5] += plaq * 0.5;	    
		    }
		  }
		    
		  for(int dd = 0; dd < 6; dd++){
		  	field[dd].real() *= loop.real();
		  	field[dd].imag() *= loop.real();		  
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
		    CudaAtomicAdd(arg.field + id0 + fieldoffset + radiusoffset, aggregate[1]);
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
      __syncthreads();
    }
  }
}
}
}





	

template <bool chargeplane>
class ChromoFieldWL: Tunable{
private:
   ChromoFieldArgWL arg;
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
	if(chargeplane) kernel_ChromoFieldWL<<<tp.grid, tp.block, tp.shared_bytes, stream>>>(arg);
	else kernel_ChromoFieldMidFluxTubeWL<<<tp.grid, tp.block, tp.shared_bytes, stream>>>(arg);
}

public:
   ChromoFieldWL(Array<complexd> *wloop, Array<complexd> *plaqfield, Array<complexd> *chromofield, int Rmax, int Tmax, int nx, int ny): chromofield(chromofield){
	size = Volume();
	timesec = 0.0;
	arg.nx = nx;
	arg.ny = ny;
	arg.Rmax = Rmax;
	arg.Tmax = Tmax;
	arg.wloop = wloop->getPtr();
	arg.plaq = plaqfield->getPtr();
	field = new Array<complexd>(Device, chromofield->Size());
	arg.field = field->getPtr();
	cout << arg.wloop << '\t' << arg.plaq << '\t' << arg.field << endl;
	cout << arg.nx << '\t' << arg.ny << '\t' << arg.Rmax << '\t' << arg.Tmax << endl; 
}
   ~ChromoFieldWL(){delete field;}
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
	for(int r = 0; r < arg.Rmax; r++){
		int id = r + arg.Rmax * t;
		if((r+1)%2){
			for(int f = 0; f < fsize; f++)
			  chromofield->at(f + id * fsize) /= double(12 * size);
			if(chargeplane){
			  for(int f = 2 * plane; f < 3 * plane; f++) //Ez^2
				chromofield->at(f + id * fsize) *= 2.0; 
			  for(int f = 5 * plane; f < 6 * plane; f++) //Bz^2
				chromofield->at(f + id * fsize) *= 0.5; 
			}
			else{
			  for(int f = plane; f < 2 * plane; f++) //Ey^2
				chromofield->at(f + id * fsize) *= 2.0; 
			  for(int f = 4 * plane; f < 5 * plane; f++) //By^2
				chromofield->at(f + id * fsize) *= 0.5; 
			}
		}
		else{
			for(int f = 0; f < fsize; f++)
			  chromofield->at(f + id * fsize) /= double(6 * size);
		}
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
   double time(){	return timesec;}
   void stat(){	cout << "ChromoField:  " <<  time() << " s\t"  << bandwidth() << " GB/s\t" << flops() << " GFlops"  << endl;}
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
void CalcChromoFieldWL(Array<complexd> *wloop, Array<complexd> *plaqfield, Array<complexd> *field, int Rmax, int Tmax, int nx, int ny){
	Timer mtime;
	mtime.start(); 
	ChromoFieldWL<chargeplane> cfield(wloop, plaqfield, field, Rmax, Tmax, nx, ny);
	cfield.Run();
	cudaDevSync( );
	mtime.stop();
	cout << "Time ChromoField:  " <<  mtime.getElapsedTimeInSec() << " s"  << endl;
}




void CalcChromoFieldWL(Array<complexd> *wloop, Array<complexd> *plaqfield, Array<complexd> *field, int Rmax, int Tmax, int nx, int ny, bool chargeplane){
	if(Dirs() < 4){
		cout << "Only implemented for 4D lattice..." << endl;
		Finalize(1);
	}
	if(chargeplane) CalcChromoFieldWL<true>(wloop, plaqfield, field, Rmax, Tmax, nx, ny);
	else CalcChromoFieldWL<false>(wloop, plaqfield, field, Rmax, Tmax, nx, ny);
}



}
