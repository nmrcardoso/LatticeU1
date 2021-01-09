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


#include "random.h"
#include "staple.h"
#include "update.h"
#include "plaquette.h"

#include "multilevel.h"
#include "tune.h"

using namespace std;


namespace U1{

void polyakov(double *lat, double *poly){
	for(int i = 0; i < 2; ++i) poly[i] = 0.;
	for(int parity = 0; parity < 2; ++parity){
		#pragma omp parallel for reduction(+:poly[:2])
		for(int id = 0; id < SpatialVolume()/2; ++id){
			int x[4];
			indexEO(id, parity, x);
			double tmp = 0.;
			for(x[TDir()] = 0; x[TDir()] < Grid(TDir()); ++x[TDir()])
				tmp += lat[ indexId(x, TDir()) ];
			poly[0] += cos(tmp);
			poly[1] += sin(tmp);
		}
	}
	double norm = 1. / double(SpatialVolume());
	for(int i = 0; i < 2; ++i) poly[i] *= norm;
}










__global__ void kernel_polyakov(double *lat, complexd *poly){
    size_t id = threadIdx.x + blockDim.x * blockIdx.x;

	complexd poly0 = 0.;
	
	if( id < SpatialVolume()/2 ){
		for(int parity = 0; parity < 2; ++parity){		
			int x[4];
			indexEO(id, parity, x);
			double tmp = 0.;
			for(x[TDir()] = 0; x[TDir()] < Grid(TDir()); ++x[TDir()])
				tmp += lat[ indexId(x, TDir()) ];
			poly0.real() += cos(tmp);
			poly0.imag() += sin(tmp);
		}
	}
	reduce_block_1d<complexd>(poly, poly0);
}



complexd dev_polyakov(double *dev_lat, complexd *dev_poly, int threads, int blocks){
	complexd poly;
	cudaSafeCall(cudaMemset(dev_poly, 0, sizeof(complexd)));
	kernel_polyakov<<<blocks, threads, threads*sizeof(complexd)>>>(dev_lat, dev_poly);
	cudaSafeCall(cudaMemcpy(&poly, dev_poly, sizeof(complexd), cudaMemcpyDeviceToHost));
	poly /= double(SpatialVolume());
	cout << "\t\t" << "L: " << poly.real() << '\t' << poly.imag() << "\t|L|: " << poly.abs() << endl;
	return poly;
} 


using namespace U1;

class Polyakov: Tunable{
private:
	double* lat;
	complexd poly;
	complexd *dev_poly;
	double norm;
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
	cudaSafeCall(cudaMemset(dev_poly, 0, sizeof(complexd)));
	kernel_polyakov<<<tp.grid, tp.block, tp.shared_bytes, stream>>>(lat, dev_poly);
}
public:
   Polyakov(double* lat) : lat(lat) {
	size = SpatialVolume()/2;
	dev_poly = (complexd*)dev_malloc(sizeof(complexd));
	norm = 1. / double(SpatialVolume());
	timesec = 0.0;  
}
   ~Polyakov(){ dev_free(dev_poly);};
   complexd Run(const cudaStream_t &stream){
#ifdef TIMMINGS
    time.start();
#endif
	apply(stream);
	cudaSafeCall(cudaMemcpy(&poly, dev_poly, sizeof(complexd), cudaMemcpyDeviceToHost));
	poly *= norm;
    CUDA_SAFE_DEVICE_SYNC();
    CUT_CHECK_ERROR("Kernel execution failed");
#ifdef TIMMINGS
	CUDA_SAFE_DEVICE_SYNC( );
    time.stop();
    timesec = time.getElapsedTimeInSec();
#endif
	return poly;
}
   complexd Run(){	return Run(0);}
   double flops(){	return ((double)flop() * 1.0e-9) / timesec;}
   double bandwidth(){	return (double)bytes() / (timesec * (double)(1 << 30));}
   long long flop() const { return 0;}
   long long bytes() const{ return 0;}
   double time(){	return timesec;}
   void stat(){	cout << "OverRelaxation:  " <<  time() << " s\t"  << bandwidth() << " GB/s\t" << flops() << " GFlops"  << endl;}
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



complexd dev_polyakov(double *dev_lat){
	complexd poly;
	Polyakov pl(dev_lat);
	poly = pl.Run();
	cout << "\t\t" << "L: " << poly.real() << '\t' << poly.imag() << "\t|L|: " << poly.abs() << endl;
	return poly;
} 










__global__ void kernel_polyakov2(double *lat, complexd *poly, int radius){
    size_t id = threadIdx.x + blockDim.x * blockIdx.x;
    
	complexd poly0 = 0.;
	
	if( id < SpatialVolume()/2 ){
		for(int parity = 0; parity < 2; ++parity){		
			int x[4];
			indexEO(id, parity, x);
			
			double tmp = 0.;
			for(x[TDir()] = 0; x[TDir()] < Grid(TDir()); ++x[TDir()])
				tmp += lat[ indexId(x, TDir()) ];
				
			for(int dir = 0; dir < TDir(); dir++){
				int xx[4];
				indexEO(id, parity, xx);
				xx[dir] = (xx[dir] + radius) % Grid(dir);
			
				double tmp1 = 0.;
				for(xx[TDir()] = 0; xx[TDir()] < Grid(TDir()); ++xx[TDir()])
					tmp1 += lat[ indexId(xx, TDir()) ];
					
					
					
					
				poly0.real() += cos(tmp-tmp1);
				poly0.imag() += sin(tmp-tmp1);
				
			}
		}
	}
	reduce_block_1d<complexd>(poly, poly0);
}



__global__ void kernel_polyakov21(double *lat, complexd *poly, int radius){
    size_t id = threadIdx.x + blockDim.x * blockIdx.x;
    
	complexd poly0 = 0.;
	
	if( id < SpatialVolume()/2 ){
		for(int parity = 0; parity < 2; ++parity){		
			int x[4];
			indexEO(id, parity, x);
			
			complexd tmp = 1.;
			for(x[TDir()] = 0; x[TDir()] < Grid(TDir()); ++x[TDir()])
				tmp *= exp_ir(lat[ indexId(x, TDir()) ]);
				
			for(int dir = 0; dir < TDir(); dir++){
				int xx[4];
				indexEO(id, parity, xx);
				xx[dir] = (xx[dir] + radius) % Grid(dir);
			
				complexd tmp1 = 1.;
				for(xx[TDir()] = 0; xx[TDir()] < Grid(TDir()); ++xx[TDir()])
					tmp1 *= exp_ir(lat[ indexId(xx, TDir()) ]);
					
					
				poly0 += tmp * conj(tmp1);
					
				
			}
		}
	}
	reduce_block_1d<complexd>(poly, poly0);
}














complexd dev_polyakov2(double *dev_lat, complexd *dev_poly, int radius, int threads, int blocks){
	complexd poly;
	cudaSafeCall(cudaMemset(dev_poly, 0, sizeof(complexd)));
	kernel_polyakov2<<<blocks, threads, threads*sizeof(complexd)>>>(dev_lat, dev_poly, radius);
	cudaSafeCall(cudaMemcpy(&poly, dev_poly, sizeof(complexd), cudaMemcpyDeviceToHost));
	poly /= double(SpatialVolume()*(Dirs()-1));
	//cout << "\t\t" << "L: " << poly.real() << '\t' << poly.imag() << "\t|L|: " << poly.abs() << endl;
	return poly;
} 









__global__ void kernel_polyakov_volume(double *lat, double *poly){
    size_t id = threadIdx.x + blockDim.x * blockIdx.x;
    
	/*if( id >= SpatialVolume()/2 ) return;
	for(int parity = 0; parity < 2; ++parity){		
		int x[4];
		indexEO(id, parity, x);
		
		double tmp = 0.;
		for(x[TDir()] = 0; x[TDir()] < Grid(TDir()); ++x[TDir()]){
			//if(id==0&&parity==0) printf("---->%d\n",x[TDir()]);
			tmp += lat[ indexId(x, TDir()) ];
		}
		poly[((x[2] * Grid(1)) + x[1] ) * Grid(0) + x[0]] = tmp;
	}
	*/
	if( id >= SpatialVolume() ) return;
	int parity = 0;
	if( id >= SpatialVolume()/2 ){
		parity = 1;	
		id -= SpatialVolume()/2;
	}	
	int x[4];
	indexEO(id, parity, x);
	
	double tmp = 0.;
	for(x[TDir()] = 0; x[TDir()] < Grid(TDir()); ++x[TDir()]){
		//if(id==0&&parity==0) printf("---->%d\n",x[TDir()]);
		tmp += lat[ indexId(x, TDir()) ];
	}
	poly[((x[2] * Grid(1)) + x[1] ) * Grid(0) + x[0]] = tmp;

}


__global__ void kernel_poly2(double *poly, complexd *poly2, int radius){
    size_t id = threadIdx.x + blockDim.x * blockIdx.x;
    
	double pl0 = 0.;
	if(id < SpatialVolume()) pl0 = poly[id];			
	int x[3];
	indexNO3D(id, x);
	for(int r = 1; r <= radius; ++r){	
		complexd pl = 0.;
		if(id < SpatialVolume()){
			for(int dir = 0; dir < TDir(); dir++){
				int xold = x[dir];
				x[dir] = (x[dir] + r) % Grid(dir);
				double pl1 = poly[((x[2] * Grid(1)) + x[1] ) * Grid(0) + x[0]];
				pl1 = pl0-pl1;
				pl.real() += cos(pl1);
				pl.imag() += sin(pl1);	
				//pl += exp_ir(pl0) * conj(exp_ir(pl1));				
				x[dir] = xold;
			}
		}				
		reduce_block_1d<complexd>(poly2 + r - 1, pl);
		__syncthreads();
	}
}

__global__ void kernel_poly21(double *poly, complexd *poly2, int radius){
    size_t id = threadIdx.x + blockDim.x * blockIdx.x;
    
	complexd pl0 = 0.;
	if(id < SpatialVolume()) pl0 = exp_ir(poly[id]);			
	int x[3];
	indexNO3D(id, x);
	for(int r = 1; r <= radius; ++r){	
		complexd pl = 0.;
		if(id < SpatialVolume()){
			for(int dir = 0; dir < TDir(); dir++){
				int xold = x[dir];
				x[dir] = (x[dir] + r) % Grid(dir);
				double pl1 = poly[((x[2] * Grid(1)) + x[1] ) * Grid(0) + x[0]];
				//pl1 = pl0-pl1;
				//pl.real() += cos(pl1);
				//pl.imag() += sin(pl1);	
				pl += pl0*conj(exp_ir(pl1));				
				x[dir] = xold;
			}
		}				
		reduce_block_1d<complexd>(poly2 + r - 1, pl);
		__syncthreads();
	}
}


complexd* poly2(double *dev_lat){
	int radius = Grid(0)/2;
	double *dev_poly_vol = (double*)dev_malloc(SpatialVolume()*sizeof(double));
	complexd *dev_poly2 = (complexd*)dev_malloc(radius*sizeof(complexd));
	complexd *poly2 = (complexd*)safe_malloc(radius*sizeof(complexd));
	
	
	int threads = 128;
	//int blocks0 = (HalfVolume() + threads - 1) / threads;
	int blocks0 = (SpatialVolume() + threads - 1) / threads;
	int blocks1 = (SpatialVolume() + threads - 1) / threads;
	size_t smem = threads * sizeof(complexd);
	
	
	kernel_polyakov_volume<<<blocks0, threads>>>(dev_lat, dev_poly_vol);
	//cudaSafeCall(cudaMemset(dev_poly2, 0, radius*sizeof(complexd)));
	kernel_poly2<<<blocks1, threads, smem>>>(dev_poly_vol, dev_poly2, radius);
	cudaSafeCall(cudaMemcpy(poly2, dev_poly2, radius*sizeof(complexd), cudaMemcpyDeviceToHost));
	
	
	

	std::ofstream fileout;
	std::string filename = "Pot_" + GetLatticeName() + ".dat";
	fileout.open (filename.c_str());
	if (!fileout.is_open()) {
		std::cout << "Error opening file: " << filename << std::endl;
		exit(1);
	}
	fileout.precision(12);
		
	for(int r = 0; r < radius; ++r){
		poly2[r] /= double(SpatialVolume()*(Dirs()-1));
		cout << r+1 << '\t' << poly2[r].real() << '\t' << poly2[r].imag() << endl;
		fileout << r+1 << '\t' << poly2[r].real() << '\t' << poly2[r].imag() << endl;
	}
	fileout.close();	
	
	dev_free(dev_poly2);
	dev_free(dev_poly_vol);
	//host_free(poly2);
	return poly2;
} 









__global__ void kernel_polyakov_volume_mhit(double *lat, complexd *poly){
    size_t id = threadIdx.x + blockDim.x * blockIdx.x;
   
	if( id >= SpatialVolume() ) return;
	int parity = 0;
	if( id >= SpatialVolume()/2 ){
		parity = 1;	
		id -= SpatialVolume()/2;
	}	
	int x[4];
	indexEO(id, parity, x);
	
	complexd res = 1.;
	for(x[TDir()] = 0; x[TDir()] < Grid(TDir()); ++x[TDir()]){
		double W_re, W_im;
		
		int pos = ((((x[3] * Grid(2) + x[2]) * Grid(1)) + x[1] ) * Grid(0) + x[0]) >> 1;
		int oddbit = (x[0] + x[1] + x[2] + x[3]) & 1;

		staple(lat, pos, oddbit, TDir(), W_re, W_im);				
		
		double alpha = sqrt(W_re*W_re+W_im*W_im);
	
		double ba = Beta() * alpha;
		double temp = cyl_bessel_i1(ba)/(cyl_bessel_i0(ba)*alpha);
		//double temp = besseli1(ba)/(besseli0(ba)*alpha);
		complexd val(temp*W_re, -temp*W_im);
		
		res *= val;
	}
	poly[((x[2] * Grid(1)) + x[1] ) * Grid(0) + x[0]] = res;

}

__global__ void kernel_poly2_mhit(complexd *poly, complexd *poly2, int radius){
    size_t id = threadIdx.x + blockDim.x * blockIdx.x;
    
	complexd pl0 = 0.;
	if(id < SpatialVolume()) pl0 = poly[id];			
	int x[3];
	indexNO3D(id, x);
	for(int r = 1; r <= radius; ++r){	
		complexd pl = 0.;
		if(id < SpatialVolume()){
			for(int dir = 0; dir < TDir(); dir++){
				int xold = x[dir];
				x[dir] = (x[dir] + r) % Grid(dir);
				complexd pl1 = poly[((x[2] * Grid(1)) + x[1] ) * Grid(0) + x[0]];
				pl += pl0 * conj(pl1);				
				x[dir] = xold;
			}
		}				
		reduce_block_1d<complexd>(poly2 + r - 1, pl);
		__syncthreads();
	}
}


complexd* poly2_mhit(double *dev_lat){
	int radius = Grid(0)/2;
	complexd *dev_poly_vol = (complexd*)dev_malloc(SpatialVolume()*sizeof(complexd));
	complexd *dev_poly2 = (complexd*)dev_malloc(radius*sizeof(complexd));
	complexd *poly2 = (complexd*)safe_malloc(radius*sizeof(complexd));
		
	int threads = 128;
	//int blocks0 = (HalfVolume() + threads - 1) / threads;
	int blocks0 = (SpatialVolume() + threads - 1) / threads;
	int blocks1 = (SpatialVolume() + threads - 1) / threads;
	size_t smem = threads * sizeof(complexd);
	
	
	kernel_polyakov_volume_mhit<<<blocks0, threads>>>(dev_lat, dev_poly_vol);
	//cudaSafeCall(cudaMemset(dev_poly2, 0, radius*sizeof(complexd)));
	kernel_poly2_mhit<<<blocks1, threads, smem>>>(dev_poly_vol, dev_poly2, radius);
	cudaSafeCall(cudaMemcpy(poly2, dev_poly2, radius*sizeof(complexd), cudaMemcpyDeviceToHost));
	
	std::ofstream fileout;
	std::string filename = "Pot_mhit_" + GetLatticeName() + ".dat";
	fileout.open (filename.c_str());
	if (!fileout.is_open()) {
		std::cout << "Error opening file: " << filename << std::endl;
		exit(1);
	}
	fileout.precision(12);
		
	for(int r = 0; r < radius; ++r){
		poly2[r] /= double(SpatialVolume()*(Dirs()-1));
		cout << r+1 << '\t' << poly2[r].real() << '\t' << poly2[r].imag() << endl;
		fileout << r+1 << '\t' << poly2[r].real() << '\t' << poly2[r].imag() << endl;
	}
	
	fileout.close();	
	
	
	
	
	//host_free(poly2);
	dev_free(dev_poly2);
	dev_free(dev_poly_vol);
	return poly2;
} 




















template< bool multihit>
__global__ void kernel_polyakov_volume(double *lat, complexd *poly){
    size_t id = threadIdx.x + blockDim.x * blockIdx.x;
   
	if( id >= SpatialVolume() ) return;
	int parity = 0;
	if( id >= SpatialVolume()/2 ){
		parity = 1;	
		id -= SpatialVolume()/2;
	}	
	int x[4];
	indexEO(id, parity, x);
	
	complexd res = 1.;
	for(x[TDir()] = 0; x[TDir()] < Grid(TDir()); ++x[TDir()]){	
		if(multihit){
			double W_re, W_im;				
			int pos = ((((x[3] * Grid(2) + x[2]) * Grid(1)) + x[1] ) * Grid(0) + x[0]) >> 1;
			int oddbit = (x[0] + x[1] + x[2] + x[3]) & 1;
			staple(lat, pos, oddbit, TDir(), W_re, W_im);			
			double alpha = sqrt(W_re*W_re+W_im*W_im);
			double ba = Beta() * alpha;
			double temp = cyl_bessel_i1(ba)/(cyl_bessel_i0(ba)*alpha);
			//double temp = besseli1(ba)/(besseli0(ba)*alpha);
			complexd val(temp*W_re, -temp*W_im);
			res *= val;
		}
		else{
			res *= exp_ir(lat[ indexId(x, TDir()) ]);
		}
	}
	poly[((x[2] * Grid(1)) + x[1] ) * Grid(0) + x[0]] = res;
}

template< bool multihit>
class Polyakov_Vol: Tunable{
private:
	double* lat;
	complexd* poly;
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
	kernel_polyakov_volume<multihit><<<tp.grid, tp.block, tp.shared_bytes, stream>>>(lat, poly);
}
public:
   Polyakov_Vol(double* lat) : lat(lat) {
	size = SpatialVolume();
	poly = (complexd*)dev_malloc(SpatialVolume()*sizeof(complexd));
	timesec = 0.0;  
}
   ~Polyakov_Vol(){ };
   complexd* Run(const cudaStream_t &stream){
#ifdef TIMMINGS
    time.start();
#endif
	apply(stream);
    CUDA_SAFE_DEVICE_SYNC();
    CUT_CHECK_ERROR("Kernel execution failed");
#ifdef TIMMINGS
	CUDA_SAFE_DEVICE_SYNC( );
    time.stop();
    timesec = time.getElapsedTimeInSec();
#endif
	return poly;
}
   complexd* Run(){	return Run(0);}
   double flops(){	return ((double)flop() * 1.0e-9) / timesec;}
   double bandwidth(){	return (double)bytes() / (timesec * (double)(1 << 30));}
   long long flop() const { return 0;}
   long long bytes() const{ return 0;}
   double time(){	return timesec;}
   void stat(){	cout << "OverRelaxation:  " <<  time() << " s\t"  << bandwidth() << " GB/s\t" << flops() << " GFlops"  << endl;}
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





__global__ void kernel_PP(complexd *poly, complexd *res, int radius){
    size_t id = threadIdx.x + blockDim.x * blockIdx.x;
    
	complexd pl0 = 0.;
	if(id < SpatialVolume()) pl0 = poly[id];			
	int x[3];
	indexNO3D(id, x);
	for(int r = 1; r <= radius; ++r){	
		complexd pl = 0.;
		if(id < SpatialVolume()){
			for(int dir = 0; dir < TDir(); dir++){
				int xold = x[dir];
				x[dir] = (x[dir] + r) % Grid(dir);
				complexd pl1 = poly[((x[2] * Grid(1)) + x[1] ) * Grid(0) + x[0]];
				pl += pl0 * conj(pl1);				
				x[dir] = xold;
			}
		}				
		reduce_block_1d<complexd>(res + r - 1, pl);
		__syncthreads();
	}
}




class PP: Tunable{
private:
	complexd* pvol;
	complexd *poly;
	complexd *dev_poly;
	int radius;
	double norm;
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
	cudaSafeCall(cudaMemset(dev_poly, 0, radius*sizeof(complexd)));
	kernel_PP<<<tp.grid, tp.block, tp.shared_bytes, stream>>>(pvol, dev_poly, radius);
}
public:
   PP(complexd *pvol, int radius) : pvol(pvol), radius(radius) {
	size = SpatialVolume();
	dev_poly = (complexd*)dev_malloc(radius*sizeof(complexd));
	poly = (complexd*)safe_malloc(radius*sizeof(complexd));
	norm = 1. / double(SpatialVolume()*(Dirs()-1));
	timesec = 0.0;  
}
   ~PP(){ dev_free(dev_poly);};
   complexd* Run(const cudaStream_t &stream){
#ifdef TIMMINGS
    time.start();
#endif
	apply(stream);
	cudaSafeCall(cudaMemcpy(poly, dev_poly, radius*sizeof(complexd), cudaMemcpyDeviceToHost));
	for(int i = 0; i < radius; ++i) poly[i] *= norm;
    CUDA_SAFE_DEVICE_SYNC();
    CUT_CHECK_ERROR("Kernel execution failed");
#ifdef TIMMINGS
	CUDA_SAFE_DEVICE_SYNC( );
    time.stop();
    timesec = time.getElapsedTimeInSec();
#endif
	return poly;
}
   complexd* Run(){	return Run(0);}
   double flops(){	return ((double)flop() * 1.0e-9) / timesec;}
   double bandwidth(){	return (double)bytes() / (timesec * (double)(1 << 30));}
   long long flop() const { return 0;}
   long long bytes() const{ return 0;}
   double time(){	return timesec;}
   void stat(){	cout << "OverRelaxation:  " <<  time() << " s\t"  << bandwidth() << " GB/s\t" << flops() << " GFlops"  << endl;}
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




complexd* Poly2(double *lat, bool multihit){
	int radius = Grid(0)/2;
	
	complexd* poly;
	if(multihit){
		Polyakov_Vol<true> pvol(lat);
		poly = pvol.Run();
	}
	else{
		Polyakov_Vol<false> pvol(lat);
		poly = pvol.Run();
	}
	PP pp(poly, radius);
	complexd* poly2 = pp.Run();
	dev_free(poly);
	
	std::ofstream fileout;
	std::string filename = "Pot_" + GetLatticeName() + ".dat";
	if(multihit) filename = "Pot_mhit_" + GetLatticeName() + ".dat";
	fileout.open (filename.c_str());
	if (!fileout.is_open()) {
		std::cout << "Error opening file: " << filename << std::endl;
		exit(1);
	}
	fileout.precision(12);
		
	for(int r = 0; r < radius; ++r){
		cout << r+1 << '\t' << poly2[r].real() << '\t' << poly2[r].imag() << endl;
		fileout << r+1 << '\t' << poly2[r].real() << '\t' << poly2[r].imag() << endl;
	}
	
	fileout.close();	
	
	//host_free(poly2);
	return poly2;
} 


}

