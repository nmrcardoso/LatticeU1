#include <iostream>
#include <math.h> 
#include <time.h> 
#include <random>
#include <vector> 
#include <fstream>
#include <omp.h>

#include "timer.h"
#include "alloc.h"
#include "complex.h"

#include "parameters.h"
#include "random.h"
#include "update.h"
#include "plaquette.h"
#include "polyakov.h"

#include "multilevel.h"


#include "array.h"

using namespace std;
using namespace U1;




int main(){
	Timer a0;
	a0.start();
	
	//omp_set_num_threads(1);
	int gpuID = 0;
	Start(gpuID); // Important. Setup GPU id and setup tune kernels and verbosity level. See cuda_error_check.cpp/.h
	
	
	int numthreads = 0;
	#pragma omp parallel
	numthreads = omp_get_num_threads();
	cout << "Number of threads: " << numthreads << endl;
	
	//create one RNG per thread
	generator = new std::mt19937[numthreads];
	for(int i = 0; i < numthreads; ++i) generator[i].seed(time(NULL)*(i+1));
	
	
	
	
	PARAMS::DIRS = 4;
	PARAMS::TDir = PARAMS::DIRS - 1;
	int ls = 24; //The number of points in each direction must be an even number!!!!!!!!!
	int Nx=ls;
	int Ny=ls;
	int Nz=ls;
	int Nt=12;
	for(int i = 0; i < 4; ++i) PARAMS::Grid[i] = 1;
	PARAMS::Grid[0] = Nx;
	if(Dirs()==2) PARAMS::Grid[1] = Nt;
	else if(Dirs() > 2) PARAMS::Grid[1] = Ny;
	if(Dirs()==3) PARAMS::Grid[2] = Nt;
	else if(Dirs() > 3) PARAMS::Grid[2] = Nz;
	if(Dirs()==4) PARAMS::Grid[3] = Nt;
	PARAMS::volume = 1;
	for(int i = 0; i < 4; ++i) PARAMS::volume *= Grid(i);	
	PARAMS::half_volume = Volume() / 2;
	PARAMS::spatial_volume = 1;
	for(int i = 0; i < TDir(); ++i) PARAMS::spatial_volume *= Grid(i);
	
	int maxIter = 1020;
	int printiter = 100;
	PARAMS::Beta = 1.;
	PARAMS::Aniso = 1.;
	bool hotstart = false;
	PARAMS::metrop = 1;
	PARAMS::ovrn = 3;
	
	
	
	int numplaqs = 6; //DIRS=4 3D+1
	if(Dirs()==2) numplaqs = 1.;
	else if(Dirs()==3) numplaqs = 3.;
	double norm = 1. / double(Volume() * numplaqs);
	
	//cout << "#####:::::: " << GetLatticeName() << endl;
	//return;
	
	//GPU Code
	Timer t0;
	t0.start();
	
	SetupGPU_Parameters(); // Copy parameters to GPU constant memory, need to be setup before any kernel call
	            

	//Array array to store the phases
	Array<double> *lattice = new Array<double>(Device, Volume()*Dirs()); //also initialize aray to 0
	//Initialize cuda rng
	CudaRNG *rng = new CudaRNG(1234, HalfVolume());
	//cuRNGState *rng_state = Init_Device_RNG(1234);
	// cuda memory container for global reductions, used in plaquette and polyakov calculations
	complexd *dev_tmp = (complexd*)dev_malloc(sizeof(complexd));

	if(hotstart){
		HotStart(lattice, rng);
	}
	cout << "Iter: " << PARAMS::iter << " \t";
	complexd plaqv = Plaquette(lattice);
	complexd ployv = Polyakov(lattice);
	
	
	string filename = "";
	for(int i = 0; i < PARAMS::DIRS; ++i) filename += ToString(PARAMS::Grid[i]) + "_";
	filename += ToString(PARAMS::Beta) + "_" + ToString(PARAMS::ovrn) + "_" + ToString(hotstart) + ".dat";
	
    ofstream fileout;
    string filename1 = "plaq_" + filename;
    fileout.open(filename1, ios::out);
    if (!fileout.is_open()) {
    	cout << "Cannot create file: " << filename1 << endl;
    	exit(1);
	}
	cout << "Creating file: " << filename1 << endl;
    fileout.precision(12);
    
    ofstream fileout1;
    filename1 = "poly_" + filename;
    fileout1.open( filename1, ios::out);
    if (!fileout1.is_open()) {
    	cout << "Cannot create file: " << filename1 << endl;
    	exit(1);
	}
	cout << "Creating file: " << filename1 << endl;
    fileout1.precision(12);
    
    fileout << PARAMS::iter << '\t' << plaqv.real() << '\t' << plaqv.imag() << endl;
    
	for(PARAMS::iter = 1; PARAMS::iter <= maxIter; ++PARAMS::iter){
		// metropolis and overrelaxation algorithm 
		UpdateLattice(lattice, rng,  PARAMS::metrop, PARAMS::ovrn);
		
		if( (PARAMS::iter%printiter)==0){
			cout << "Iter: " << PARAMS::iter << " \t";
			//plaqv = dev_plaquette(lattice.getPtr(), dev_tmp, norm, threads, blocks);
			//ployv = dev_polyakov(lattice.getPtr(), dev_tmp, threads, pblocks);
			plaqv = Plaquette(lattice);
			ployv = Polyakov(lattice);
			fileout << PARAMS::iter << '\t' << plaqv.real() << '\t' << plaqv.imag() << endl;
			fileout1 << PARAMS::iter << '\t' << ployv.real() << '\t' << ployv.imag() << '\t' << ployv.abs() << endl;
		}
		if( PARAMS::iter > 990 && (PARAMS::iter%printiter)==0){
			cout << "################################" << endl;
			Timer p0, p1, p2, p3;
			cout << "########### P(0)*conj(P(r)) #####################" << endl;
			p0.start();
			Array<complexd>* res = Poly2(lattice, false);
			delete res;
			p0.stop();
			std::cout << "p0: " << p0.getElapsedTime() << " s" << endl;	
			cout << "########### P(0)*conj(P(r)) Using MultiHit #####################" << endl;
			p1.start();
			res = Poly2(lattice, true);
			delete res;
			p1.stop();
			std::cout << "p1: " << p1.getElapsedTime() << " s" << endl;			
			cout << "########### P(0)*conj(P(r)) Using MultiLevel #####################" << endl;
			p2.start();
			//MultiLevel(lattice.getPtr(), rng.getPtr(), 50, 10, 50, 10, 2, 5);
			//MultiLevel(lattice.getPtr(), rng.getPtr(), 1, 1, 1, 1, 1, 3);
			//res = MultiLevel(lattice.getPtr(), rng.getPtr(), 1, 0, 1, 0, 1, 3);
			Array<complexd>* results = MultiLevel(lattice, rng, 10, 16, 25, 5, 2, 5);
			delete results;
			p2.stop();
			std::cout << "p2: " << p2.getElapsedTime() << " s" << endl;
			cout << "################################" << endl;			
		}			
	}
	fileout.close();
	fileout1.close();
	
	
	dev_free(dev_tmp);
	delete lattice;
	delete rng;
	t0.stop();
	std::cout << "Time: " << t0.getElapsedTime() << " s" << endl;
	
	Finalize(0); // Important to save tunned kernels to file
	return 0;
/*	
	//CPU CODE
	Timer t1;
	t1.start();
	// creates the lattice array and initializes it to 0, cold start
	double *lat = new double[Volume()*Dirs()](); 
	
	std::uniform_real_distribution<double> rand02(0., 2.);
	std::uniform_real_distribution<double> rand01(0,1);
	
	PARAMS::iter = 0;
	if(hotstart) {
		//Initializes lattice array with random phase (hot start) between 0-2Pi
		#pragma omp parallel for	
		for(int id = 0; id < Volume()*Dirs(); ++id) 
			lat[id] = M_PI * rand02(generator[omp_get_thread_num()]);
	}
	double plaq[2];
	double poly[2];
	plaquette(lat, plaq);
	polyakov(lat, poly);
	cout << "iter: " << PARAMS::iter << " \tplaq: " << 1.-plaq[0] << '\t' << plaq[1] << endl;
	cout << "           " << " \tL: " << poly[0] << '\t' << poly[1] << "\t|L|: " << sqrt(poly[0]*poly[0]+poly[1]*poly[1]) << endl;
	for(PARAMS::iter = 1; PARAMS::iter <= maxIter; ++PARAMS::iter){
		metropolis(lat);
		for(int ovr = 0; ovr < ovrn; ++ovr)
			overrelaxation(lat);
		if( (PARAMS::iter%printiter)==0){
			plaquette(lat, plaq);
			polyakov(lat, poly);
			cout << "iter: " << PARAMS::iter << " \tplaq: " << 1.-plaq[0] << '\t' << plaq[1] << endl;
			cout << "           " << " \tL: " << poly[0] << '\t' << poly[1] << "\t|L|: " << sqrt(poly[0]*poly[0]+poly[1]*poly[1]) << endl;
		}
	}
	t1.stop();
	std::cout << "Time: " << t1.getElapsedTime() << endl;
	std::cout << "SpeeUp: " << t1.getElapsedTime()/t0.getElapsedTime() << endl;
	cout << "Acceptation ratio: " << PARAMS::accept_ratio/double(Volume()*Dirs()*PARAMS::iter) << endl;
	
	delete[] lat;
	delete[] generator;	
	a0.stop();
	std::cout << "Time: " << a0.getElapsedTime() << endl;
	Finalize(0);
	return 0;*/
	
}
