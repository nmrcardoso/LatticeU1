#include <cstdlib>
#include <cstdio>
#include <string>
#include <cstring>
#include <map>
#include <unistd.h> // for getpagesize()
#include <execinfo.h> // for backtrace
#include <iostream>
#include <vector>


namespace U1{


void calculateCorTime(int miniter, int iter, std::vector<double> &gamma, int &nsweep){
	//Timer a0; a0.start();
	static bool calc =  false;
	if(gamma.size() < miniter) return;
	if(calc) return;
	double avg = 0.;
	for(int i = 0; i < gamma.size(); ++i) avg += gamma[i];
	avg /= double(gamma.size());

	double rho=0.;
	for(int i = 0; i < gamma.size(); ++i){
		double tmp = gamma[i] - avg;
		rho += tmp * tmp;
	}
	rho /= double(gamma.size());

	std::vector<double> corr;
	for(int j = 0; j < gamma.size(); ++j){
		double ga = 0.;
		for(int i = 0; i < gamma.size()-j; ++i)
			ga += (gamma[i] - avg) * (gamma[i+j] - avg);
		ga /= double(gamma.size()-j);
		corr.push_back(ga/rho);
	}

	double tau_int = 0.5;
	for(int i = 1; i < gamma.size()-1;++i){
		tau_int += corr[i-1];
		int tauInt = int(4.*tau_int+1);
		if( i > tauInt ) {
			nsweep = i;
			std::cout << "iter: " << iter << "\tsweeps: " << nsweep << "\ttau_int: " << tau_int << "\tint(4.*tau_int+1): " << tauInt << std::endl;
			gamma.clear();
			//calc = true;
			break;
		}
	}	
	//a0.stop();
	//std::cout << "AC Time: " << a0.getElapsedTime() << " s" << std::endl;
	std::cout << "-----------------------------------------" << std::endl;
}



}



