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
#include "complex.h"

#include "parameters.h"
#include "random.h"
#include "plaquette.h"


#include "fields.h"
#include "multilevel.h"
#include "polyakov.h"
#include "wilsonloop.h"


using namespace std;


namespace U1{








Array<complexd>* GetPPFields(Array<complexd> *pp, Array<complexd> *ppfield, Array<complexd> *plaqf, Array<complexd> *plaqfield, int Rmax, bool chargeplane){
	int nx = Grid(0);
	int ny = nx;
	double Rmin = 2; //Only makes sense to calculate for r >= 2
	Array<complexd> *field = new Array<complexd>(Host, 6 * nx * ny * Rmax);
	CalcChromoFieldPP(ppfield, plaqfield, field, Rmin, Rmax, nx, ny, chargeplane);
	//cout << ppfield->at(0) << ":" << plaqfield->at(0) << ":" << field->at(0) <<endl;
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

void Calc_PPFields(Array<double>* lattice, CudaRNG *rng, bool chargeplane, bool multilevel){
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
	
	if(multilevel) {
		//DON'T FORGET TO TUNE THE MULTIVEL!!!!!!!!!!!!!		
		MLArg arg;
		arg.Rmax() = Rmax;
		arg.nLinksLvl0() = 2;
		arg.StepsLvl0() = 50;
		arg.UpdatesLvl0() = 5;
		arg.nLinksLvl1() = 4;
		arg.StepsLvl1() = 10;
		arg.UpdatesLvl1() = 16;
		arg.nUpdatesMetropolis() = 1;
		arg.nUpdatesOvr() = 3;
		arg.MHit() = true;
		auto data = MultiLevelField(lattice, rng, &arg);
		pp = std::get<0>(data);
		ppfield = std::get<1>(data);
	}
	else{
		bool multihit = true;
		Poly2(lattice, &pp, &ppfield, true);
	}
	Array<complexd> *field0 = GetPPFields(pp, ppfield, plaqf, plaqfield, Rmax, chargeplane);
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

void Calc_WLFields(Array<double>* lattice, CudaRNG *rng, bool chargeplane){
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
		
	Array<complexd> *field0 = GetWLFields(wl, wlfield, plaqf, plaqfield, Rmax, Tmax, chargeplane);
	delete field0;
	delete plaqfield;
	delete plaqf;
	delete wl;
	delete wlfield;
}




}
