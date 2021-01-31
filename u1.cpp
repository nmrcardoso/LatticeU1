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


#include "chromofield.h"
#include "smearing.h"
#include "wilsonloop.h"


using namespace std;
using namespace U1;


Array<complexd>* GetPPFields(Array<complexd> *pp, Array<complexd> *ppfield, Array<complexd> *plaqf, Array<complexd> *plaqfield, int Rmax, bool chargeplane){
	int nx = Grid(0);
	int ny = nx;
	double Rmin = 2; //Only makes sense to calculate for r >= 2
	Array<complexd> *field = new Array<complexd>(Host, 6 * nx * ny * Rmax);
	CalcChromoFieldPP(ppfield, plaqfield, field, Rmin, Rmax, nx, ny, chargeplane);
	int plane = nx * ny;
	int fsize = 6 * plane;
	for(int r = Rmin; r < Rmax; r++){
		ofstream fieldsout;
		string fname = "ChromoField_PP_";
		if(!chargeplane) fname = "ChromoField_PP_mid_";
		fname += GetLatticeNameI() + "_r_" + ToString(r) + ".dat";;
		fieldsout.open(fname, ios::out);
		if (!fieldsout.is_open()) {
			cout << "Cannot create file: " << fname << endl;
			exit(1);
		}
		cout << "Saving data to " << fname << endl;
		fieldsout << std::scientific;
		fieldsout.precision(14);
		fieldsout << nx << '\t' << ny << '\t' << r+1 << endl;
		fieldsout << pp->at(r) << endl;
		for(int f = 0; f < 6; f++)
			fieldsout << plaqf->at(f) << endl;
		
				
		for( int ix = 0; ix < nx; ++ix )
		for( int iy = 0; iy < ny; ++iy ) {
			int id0 = ix + nx * iy;
			fieldsout << ix - nx/2 << '\t' << iy - ny/2;
			for(int f = 0; f < 6; f++){
				int id1 = id0 + f * plane + fsize * r;
				fieldsout << '\t' << field->at(id1);
			}
			fieldsout << endl;
		}
		fieldsout.close();
	}
	return field;
}

void Calc_PPFields(Array<double>* lattice, CudaRNG *rng){
	complexd plaqv[2];
	Plaquette(lattice, plaqv, true);
	Array<double>* latno = LatticeConvert(lattice, true);

	int Rmax = Grid(0)/2+1;
	Array<complexd> *plaqfield; //This array is allocated and filled in PlaquetteFields
	Array<complexd> *plaqf; //This array is allocated and filled in PlaquetteFields
	PlaquetteFields(latno, &plaqfield, &plaqf, false, false);
	delete latno;
	Array<complexd> *pp; //This array is allocated and filled in MultiLevelField
	Array<complexd> *ppfield; //This array is allocated and filled in MultiLevelField
	
	bool multilevel = true;
	if(multilevel) {
		//DON'T FORGET TO TUNE THE MULTIVEL!!!!!!!!!!!!!		
		MultiLevelField(lattice, rng, &pp, &ppfield, 1, 0, 1, 0, 1, 3, Rmax);
		//MultiLevelField(lattice, rng, &pp, &ppfield, 10, 16, 100, 5, 2, 5);
	}
	else{
		bool multihit = true;
		Poly2(lattice, &pp, &ppfield, true);
	}
	Array<complexd> *field0 = GetPPFields(pp, ppfield, plaqf, plaqfield, Rmax, true);
	delete field0;
	delete plaqfield;
	delete plaqf;
	delete pp;
	delete ppfield;
}






Array<complexd>* GetWLFields(Array<complexd> *wl, Array<complexd> *wlfield, Array<complexd> *plaqf, Array<complexd> *plaqfield, int Rmax, int Tmax, bool chargeplane){
	int nx = Grid(0);
	int ny = nx;
	double Rmin = 2; //Only makes sense to calculate for r >= 2
	Array<complexd> *field = new Array<complexd>(Host, 6 * nx * ny * Rmax * Tmax);
	CalcChromoFieldWL(wlfield, plaqfield, field, Rmin, Rmax, Tmax, nx, ny, chargeplane);
	int plane = nx * ny;
	int fsize = 6 * plane;
	for(int r = Rmin; r < Rmax; r++){
		ofstream fieldsout;
		string fname = "ChromoField_WL_";
		if(!chargeplane) fname = "ChromoField_WL_mid_";
		fname += GetLatticeNameI() + "_r_" + ToString(r) + ".dat";;
		fieldsout.open(fname, ios::out);
		if (!fieldsout.is_open()) {
			cout << "Cannot create file: " << fname << endl;
			exit(1);
		}
		cout << "Saving data to " << fname << endl;
		fieldsout << std::scientific;
		fieldsout.precision(14);
		fieldsout << nx << '\t' << ny << '\t' << r << '\t' << Tmax << endl;
		for(int t = 0; t < Tmax; t++){
			fieldsout << wl->at(t+r*Tmax) << endl;
		}
		for(int f = 0; f < 6; f++)
			fieldsout << plaqf->at(f) << endl;
							
		for( int ix = 0; ix < nx; ++ix )
		for( int iy = 0; iy < ny; ++iy ) {
			int id0 = ix + nx * iy;
			for(int t = 0; t < Tmax; t++){
				int idr = r + Rmax * t;	
				fieldsout << ix - nx/2 << '\t' << iy - ny/2 << '\t' << t;
				for(int f = 0; f < 6; f++){
					int id1 = id0 + f * plane + fsize * idr;
					fieldsout << '\t' << field->at(id1);
				}
				fieldsout << endl;
			}
		}
		fieldsout.close();
	}
	return field;
}

void Calc_WLFields(Array<double>* lattice, CudaRNG *rng){
	cout << "------------------------------" << endl;	
	Array<complexd> *plaqfield; //This array is allocated and filled in PlaquetteFields
	Array<complexd> *plaqf; //This array is allocated and filled in PlaquetteFields
	PlaquetteFields(lattice, &plaqfield, &plaqf, true, true);
	
	int Rmax = Grid(0)/2+1;
	int Tmax = Grid(TDir())/2+1;
	Array<complexd> *wl;
	Array<complexd> *wlfield;
	WilsonLoop(lattice, &wl, &wlfield, Rmax, Tmax, true);
	for(int r = 0; r < Rmax; r++)
	for(int t = 0; t < Tmax; t++)
		cout << r << '\t' << t << '\t' << wl->at(t+r*Tmax) << endl;
		
	Array<complexd> *field0 = GetWLFields(wl, wlfield, plaqf, plaqfield, Rmax, Tmax, true);
	delete field0;
	delete plaqfield;
	delete plaqf;
	delete wl;
	delete wlfield;
}




int main(){
	Timer a0;
	a0.start();
	
	//omp_set_num_threads(1);
	int gpuID = 0;
	Start(gpuID, DEBUG_VERBOSE, TUNE_YES); 
	//Start(gpuID, VERBOSE, TUNE_YES); // Important. Setup GPU id and setup tune kernels and verbosity level. See cuda_error_check.cpp/.h
	
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
	
	
	
	/*Array<double> *aa = new Array<double>(Device, Volume()*Dirs()); 
	
	double *ptr = 0;
	cout << ptr << endl;
	aa->Allocate(&ptr, Host, 10);
	cout << ptr << endl;
	cout << ptr[0] << endl;
	cout << ptr[1] << endl;
	aa->Release(ptr, Host);
	cout << ptr << endl;
	
	
	return 0;*/
	
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
    
   
    //Only used if: export U1_ENABLE_MANAGED_MEMORY=1
    use_managed_memory();

	//Array array to store the phases
	Array<double> *lattice = new Array<double>(Device, Volume()*Dirs()); //also initialize aray to 0
	//Array<double> *lattice = new Array<double>(Managed, Volume()*Dirs()); //also initialize aray to 0
	//cout << "<----->: " << (*lattice)(0) << endl;
	
	//Initialize cuda rng
	int seed = 1234;
	CudaRNG *rng = new CudaRNG(seed, HalfVolume());
	//CudaRNG1 *rngv = new CudaRNG1(seed, HalfVolume());
	
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
	vector<Array<complexd>*> data;
	for(PARAMS::iter = 1; PARAMS::iter <= maxIter; ++PARAMS::iter){
		// metropolis and overrelaxation algorithm 
		UpdateLattice(lattice, rng,  PARAMS::metrop, PARAMS::ovrn);
		//UpdateLattice(lattice, rngv, PARAMS::metrop, PARAMS::ovrn);
		
		if((PARAMS::iter%printiter)==0){
			Plaquette(lattice, plaqv);
			cout << PARAMS::iter << '\t' << plaqv[0] << '\t' << plaqv[1] << endl;
			fileout << PARAMS::iter << '\t' << plaqv[0] << '\t' << plaqv[1] << endl;
			ployv = Polyakov(lattice);
			cout << "L: " << ployv.real() << '\t' << ployv.imag() << "\t|L|: " << ployv.abs() << endl;
			fileout1 << PARAMS::iter << '\t' << ployv.real() << '\t' << ployv.imag() << "\t|L|: " << ployv.abs() << endl;
		}
		if(0)if( PARAMS::iter >= 1000 && (PARAMS::iter%printiter)==0){
			Array<complexd> *fmunu_vol, *fmunu;
			Fmunu(lattice, &fmunu_vol, &fmunu);
			delete fmunu_vol;
			delete fmunu;
		}
		
		
		if(1)if( PARAMS::iter >= 1000 && (PARAMS::iter%printiter)==0){
		
			Calc_PPFields(lattice, rng);
			//Calc_WLFields(lattice, rng);
			break;
			Timer p0,p1;
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
			int Rmax = Grid(0)/2+1;
			CudaRNG *rng11 = new CudaRNG(seed, HalfVolume());
			Array<complexd>* rresults = MLgeneric::MultiLevel(lattice, rng11, 2, 4, 5, 10, 20, 5, 1, 3, Rmax, false);
			delete rng11; delete rresults;
			rng11 = new CudaRNG(seed, HalfVolume());
			rresults = MultiLevel(lattice, rng11, 5, 10, 20, 5, 1, 3, Rmax, false);
			delete rng11; delete rresults;
		}
		
			
		if(1)if( PARAMS::iter >= 1000 && (PARAMS::iter%printiter)==0){
			if(0){
				cout << "########### P(0)*conj(P(r))O_munu Using MultiLevel #####################" << endl;
				Timer p2;
				p2.start();
				int radius = 8;
				CudaRNG *rng11 = new CudaRNG(seed, HalfVolume());
				for(int radius = 2; radius <= 8; radius++){
					bool SquaredField = true;
					bool alongCharges = false; 
					bool symmetrize = false;
					int2 perpPoint = make_int2(0,0);
					Array<complexd>* res0 = MultiLevelTTO(lattice, rng11, 5, 16, 5, 5, 2, 5, radius, SquaredField, alongCharges, symmetrize, perpPoint);
					delete res0;
				}
				delete rng11;
				p2.stop();
				std::cout << "p2: " << p2.getElapsedTime() << " s" << endl;
				cout << "################################" << endl;	
			}
			if(0){
				Timer p2;p2.start();
				int radius = 8;
				CudaRNG *rng11 = new CudaRNG(seed, HalfVolume());
				for(int radius = 2; radius <= 8; radius++){
					bool SquaredField = true;
					bool alongCharges = false; 
					bool symmetrize = false;
					int2 perpPoint = make_int2(0,0);
					Array<complexd>* res0 = ML_TTO_generic::MultiLevelTTO(lattice, rng11, 2, 4, 5, 16, 5, 5, 2, 5, radius, SquaredField, alongCharges, symmetrize, perpPoint);
					delete res0;
				}
				delete rng11;
				p2.stop();
				std::cout << "p2: " << p2.getElapsedTime() << " s" << endl;
				cout << "################################" << endl;	
			}
			
		
			if(0){
				cout << "########### P(0)*conj(P(r)) Using MultiLevel #####################" << endl;
				int Rmax = Grid(0)/2;
				CudaRNG *rng11 = new CudaRNG(seed, HalfVolume());
				Array<complexd>* results;
				Array<complexd> *pp;
				Array<complexd>*ppfield;
				//results = MultiLevel(lattice, rng, 1000, 20, 200, 5, 1, 3, Rmax, false);
				results = MultiLevel(lattice, rng11, 2, 20, 20, 5, 1, 3, Rmax, false);
				//MultiLevelField(lattice, rng11, &pp, &ppfield, 2, 20, 20, 5, 1, 3, Rmax, false);
				delete rng11;
				
				cout << "################################" << endl;	
				rng11 = new CudaRNG(seed, HalfVolume());
				Array<complexd>* rresults = MLgeneric::MultiLevel(lattice, rng11, 2, 4, 2, 20, 20, 5, 1, 3, Rmax, false);
				delete rng11;
				break;
				data.push_back(results);
				//delete results;
				cout << "################################" << endl;	
				cout << "############ MEAN ##############" << endl;
				Array<double> mean( Host, results->Size());
				for(int i = 0; i < data.size(); i++){
					Array<complexd>* val = data.at(i);
					for(int r = 0; r < mean.Size(); r++) 
						mean(r) += val->at(r).real();
				}
				
				vector<double> *trials = new vector<double>[results->Size()];
				for(int i = 0; i < data.size(); i++){
					Array<complexd>* val = data.at(i);
					for(int r = 0; r < mean.Size(); r++) {
						double vr = -log(mean(r)-val->at(r).real())/double(Grid(TDir()));
						trials[r].push_back(vr);
					}
				}			
				
				for(int r = 0; r < mean.Size(); r++){ 
					mean(r) /= double(data.size());
					mean(r) = -log(mean(r))/double(Grid(TDir()));
				}
					
				double jackfactor = double(data.size()-1)/double(data.size());
				Array<double> jackerr( Host, results->Size());
				for(int r = 0; r < mean.Size(); r++){
					double jackmean = 0.0;
					for(int i = 0; i < trials[r].size(); i++){
						jackmean += trials[r][i];//.at(i);
					}
					jackmean /= double(trials[r].size());
					double jack = 0.0;
					for(int i = 0; i < trials[r].size(); i++){
						double tmp = trials[r][i] - jackmean;
						jack += tmp * tmp;
					}
					jackerr(r) = sqrt(jack * jackfactor);
				}	
					
				cout << "# of configs: " << data.size() << endl;
				for(int r = 0; r < mean.Size(); r++) {
					cout << r+1 << '\t' << mean(r) << '\t' << jackerr(r) << endl;			
				}
				cout << "################################" << endl;	
			}
		}
		
		
		
		


		if(0)if( PARAMS::iter >= 1000 && (PARAMS::iter%printiter)==0){
		
			cout << "################################" << endl;
			Timer p0, p1, p2, p3;
			if(1){
				cout << "########### Wilson Loop #####################" << endl;
				p0.start();
				int Rmax = Grid(0)/2+1;
				int Tmax = Grid(TDir())/2+1;
				Array<complexd>* wlres = WilsonLoop(lattice, Rmax, Tmax, true);
				for(int r = 0; r < Rmax; r++)
				for(int t = 0; t < Tmax; t++)
					cout << r << '\t' << t << '\t' << wlres->at(t+r*Tmax) << endl;
				delete wlres;
				std::cout << "p0: " << p0.getElapsedTime() << " s" << endl;	
			}
			if(1){
				cout << "########### Wilson Loop #####################" << endl;
				p0.start();
				int Rmax = Grid(0)/2+1;
				int Tmax = Grid(TDir())/2+1;
				Array<complexd>* wlres = WilsonLoop(lattice, Rmax, Tmax, false);
				for(int r = 0; r < Rmax; r++)
				for(int t = 0; t < Tmax; t++)
					cout << r << '\t' << t << '\t' << wlres->at(t+r*Tmax) << endl;
				delete wlres;
				std::cout << "p0: " << p0.getElapsedTime() << " s" << endl;	
			}
		/*	cout << "########### Wilson Loop with MultiHit #####################" << endl;
			p0.start();
			Array<complexd>* out11 = ApplyMultiHit(lattice, 1);			
			wlres = WilsonLoop(out11, Rmax, Tmax);
			delete out11;
			for(int r = 0; r < Rmax; r++)
			for(int t = 0; t < Tmax; t++)
				cout << r << '\t' << t << '\t' << wlres->at(t+r*Tmax) << endl;
			delete wlres;
			std::cout << "p0: " << p0.getElapsedTime() << " s" << endl;	
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
			cout << "########### P(0)*conj(P(r)) Using MultiHit a #####################" << endl;
			p1.start();			
			Array<complexd>* out = ApplyMultiHit(lattice, 1);
			res = Poly2(out, false);
			delete out;
			delete res;
			p1.stop();
			std::cout << "p1: " << p1.getElapsedTime() << " s" << endl;	
			cout << "########### P(0)*conj(P(r)) Using APE time #####################" << endl;
			p1.start();
			Array<double>* out1 = ApplyAPE(lattice, .9, 1, 1);
			res = Poly2(out1, false);
			delete out1;
			delete res;
			p1.stop();
			std::cout << "p1: " << p1.getElapsedTime() << " s" << endl;			
			cout << "########### P(0)*conj(P(r)) Using MultiLevel #####################" << endl;
			p2.start();
			int Rmax = Grid(0)/2;
			//MultiLevel(lattice, rng, 50, 10, 50, 10, 2, 5, Rmax);
			//MultiLevel(lattice, rng, 1, 1, 1, 1, 1, 3, Rmax);
			//Array<complexd>* results = MultiLevel(lattice, rng, 1, 0, 1, 0, 1, 3, Rmax);
			
			CudaRNG *rng1 = new CudaRNG(1234, HalfVolume());
			
			//Array<complexd>* results = MultiLevel(lattice, rng1, 10, 16, 25, 5, 2, 5, Rmax);
			Array<complexd>* results = MultiLevel(lattice, rng1, 10, 1, 20, 1, 2, 5, Rmax);
			delete results;
			p2.stop();
			std::cout << "p2: " << p2.getElapsedTime() << " s" << endl;
			cout << "################################" << endl;		
			
			delete rng1;
			*/
			
			
						
			if(0){
				cout << "########### P(0)*conj(P(r)) Using MultiLevel #####################" << endl;
				p2.start();
				int Rmax = Grid(0)/2;
				Array<complexd>* results = MultiLevel(lattice, rng, 1000, 20, 200, 5, 1, 3, Rmax, true);
				delete results;
				p2.stop();
				std::cout << "p2: " << p2.getElapsedTime() << " s" << endl;
				cout << "################################" << endl;	
			}
			
			
			
			
			if(0){
				cout << "########### P(0)*conj(P(r))O_munu Using MultiLevel #####################" << endl;
				p2.start();
				int radius = 8;
				for(int radius = 2; radius <= 8; radius++){
					bool SquaredField = true;
					bool alongCharges = false; 
					bool symmetrize = false;
					int2 perpPoint = make_int2(0,0);
					Array<complexd>* res0 = MultiLevelTTO(lattice, rng, 10, 16, 50, 5, 2, 5, radius, SquaredField, alongCharges, symmetrize, perpPoint);
					delete res0;
				}
				p2.stop();
				std::cout << "p2: " << p2.getElapsedTime() << " s" << endl;
				cout << "################################" << endl;	
			}
			//break;
		}


		if(0)
		if(PARAMS::iter >= mininter){
			Plaquette(lattice, plaqv, false);
			plaq_corr.push_back( (plaqv[0].real()+plaqv[1].real())*0.5);
				
			int nsweep = 0;
			//calculateCorTime(mininter+100, PARAMS::iter, plaq_corr, nsweep);
			calculateCorTime(5, PARAMS::iter, plaq_corr, nsweep);
			calculateCorTime1(5, PARAMS::iter, plaq_corr, nsweep);
		}
		
		if(0)if( PARAMS::iter >= 1000 && (PARAMS::iter%printiter)==0){
			//Calc_WLFields(lattice, rng);
			Calc_PPFields(lattice, rng);
		}
		
		if(0&&(PARAMS::iter%printiter)==0){
			cout << "Iter: " << PARAMS::iter << " \t";
			//Plaquette(lattice, plaqv);
			ployv = Polyakov(lattice);
			fileout << PARAMS::iter << '\t' << plaqv[0] << '\t' << plaqv[1] << endl;
			fileout1 << PARAMS::iter << '\t' << ployv.real() << '\t' << ployv.imag() << '\t' << ployv.abs() << endl;			
		}			
	}
	fileout.close();
	fileout1.close();
	for(int i = 0; i < data.size(); i++){
		Array<complexd>* val = data.at(i);
		delete val;
	}
	
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

