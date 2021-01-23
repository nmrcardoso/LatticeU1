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
#include "actime.h"


using namespace std;
using namespace U1;










int main(){
	Timer a0;
	a0.start();
	
	//omp_set_num_threads(1);
	int gpuID = 0;
	Start(gpuID, VERBOSE, TUNE_YES); // Important. Setup GPU id and setup tune kernels and verbosity level. See cuda_error_check.cpp/.h
	
	int dirs = 4; //Need to update kernels to take into account less than 4 directions
	int ls = 24; //The number of points in each direction must be an even number!!!!!!!!!
	int Nx=ls;
	int Ny=ls;
	int Nz=ls;
	int Nt=12;
	double beta = 1.;
	double aniso = 1.;
	int imetrop = 1;
	int ovrn = 3;
	
	//Setup global parameters in Host and Device
	SetupLatticeParameters(Nx, Ny, Nz, Nt, dirs, beta, aniso, imetrop, ovrn);
	
	int maxIter = 1000;
	int printiter = 100;
	bool hotstart = false;
	
	//GPU Code
	Timer t0;
	t0.start();
	
	
	//cout << "#####:::::: " << GetLatticeName() << endl;
      
	string filename = GetLatticeName() + ".dat";
	
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
    
    
    
    
    

	//Array array to store the phases
	Array<double> *lattice = new Array<double>(Device, Volume()*Dirs()); //also initialize aray to 0
	//Initialize cuda rng
	int seed = 1234;
	CudaRNG *rng = new CudaRNG(seed, HalfVolume());
	if(hotstart){
		HotStart(lattice, rng);
	}
	cout << "Iter: " << PARAMS::iter << " \t";
	complexd *plaqv = new complexd[2];
	Plaquette(lattice, plaqv, false);
	complexd ployv = Polyakov(lattice);
	cout << PARAMS::iter << '\t' << plaqv[0] << '\t' << plaqv[1] << endl;
	cout << "L: " << ployv.real() << '\t' << ployv.imag() << "\t|L|: " << ployv.abs() << endl;
    fileout << PARAMS::iter << '\t' << plaqv[0] << '\t' << plaqv[1] << endl;
	fileout1 << PARAMS::iter << '\t' << ployv.real() << '\t' << ployv.imag() << "\t|L|: " << ployv.abs() << endl;
  
      
    vector<double> plaq_corr;
	int mininter = 700;
	for(PARAMS::iter = 1; PARAMS::iter <= maxIter; ++PARAMS::iter){
		// metropolis and overrelaxation algorithm 
		UpdateLattice(lattice, rng,  PARAMS::metrop, PARAMS::ovrn);
		
		if((PARAMS::iter%printiter)==0){
			Plaquette(lattice, plaqv);
			cout << PARAMS::iter << '\t' << plaqv[0] << '\t' << plaqv[1] << endl;
			fileout << PARAMS::iter << '\t' << plaqv[0] << '\t' << plaqv[1] << endl;
			ployv = Polyakov(lattice);
			cout << "L: " << ployv.real() << '\t' << ployv.imag() << "\t|L|: " << ployv.abs() << endl;
			fileout1 << PARAMS::iter << '\t' << ployv.real() << '\t' << ployv.imag() << "\t|L|: " << ployv.abs() << endl;
		}


		if( PARAMS::iter >= 1000 && (PARAMS::iter%printiter)==0){
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
			Array<complexd>* results = MultiLevel(lattice, rng, 10, 1, 20, 1, 2, 5);
			delete results;
			p2.stop();
			std::cout << "p2: " << p2.getElapsedTime() << " s" << endl;
			cout << "################################" << endl;		
		}


		if(0) if(PARAMS::iter >= mininter){
			Plaquette(lattice, plaqv, false);
			plaq_corr.push_back( (plaqv[0].real()+plaqv[1].real())*0.5);
				
			int nsweep = 0;
			//calculateCorTime(mininter+100, PARAMS::iter, plaq_corr, nsweep);
			calculateCorTime(5, PARAMS::iter, plaq_corr, nsweep);
			calculateCorTime1(5, PARAMS::iter, plaq_corr, nsweep);
		}
	}
	fileout.close();
	fileout1.close();
	
	delete lattice;
	delete rng;
	delete[] plaqv;
	t0.stop();
	std::cout << "Time: " << t0.getElapsedTime() << " s" << endl;
	
	Finalize(0); // Important to save tunned kernels to file
	return 0;
/*	
	//CPU CODE
	Timer t1;
	t1.start();
	
	
	
	int numthreads = 0;
	#pragma omp parallel
	numthreads = omp_get_num_threads();
	cout << "Number of threads: " << numthreads << endl;
	
	//create one RNG per thread
	generator = new std::mt19937[numthreads];
	for(int i = 0; i < numthreads; ++i) generator[i].seed(time(NULL)*(i+1));
	
	
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

