#include <cstdlib>
#include <cstdio>
#include <string>
#include <cstring>
#include <map>
#include <unistd.h> // for getpagesize()
#include <execinfo.h> // for backtrace
#include <iostream>
#include <vector>
#include <math.h>


namespace U1{




// From paper https://arxiv.org/abs/hep-lat/0306017
void calculateCorTime(int miniter, int iter, std::vector<double> &gamma, int &nsweep){
	using namespace std;
	//Timer a0; a0.start();
	static bool calc =  false;
	if(gamma.size() < miniter) return;
	if(calc) return;
	double avg = 0.;
	for(int i = 0; i < gamma.size(); ++i) avg += gamma[i];
	avg /= double(gamma.size());


	int N = gamma.size();
	int Wopt = 0;
	int tmax = N / 2;



	
	std::vector<double> GammaFbb;
	//std::vector<double> corr;
	double rho=0.;
	double Gint = 0.;
	double tauW = 0.;
	double Stau = 1.5;
	double gW = 0.;
	bool flag = true;
	int t = 0;
	while( t <= tmax ) {
		/*double g0 = 0.;
		double gt = 0.;
		for(int i = 0; i < N-t; ++i){
			g0 += gamma[i];
			gt += gamma[i+t];
		}
		double norm = 1.0 / double(N-t);
		g0 *= norm;
		gt /= norm;
		double ga = 0.;
		for(int i = 0; i < N-t; ++i){
			ga += (gamma[i] - g0) * (gamma[i+t] - gt);
		}
		ga *= norm;*/
		
		double ga = 0.;
		for(int i = 0; i < N-t; ++i)
			ga += (gamma[i] - avg) * (gamma[i+t] - avg);
		ga /= double(N-t);
		
		if(t == 0) rho = ga;//std::cout << rho << '\t' << ga << '\t' << ga/rho << std::endl;
		
		GammaFbb.push_back(ga);
		//corr.push_back(ga/rho);
		//if(t==0) cout << "GammaFbb[0]: " << ga << endl;
		if( t > 0 )
		if(flag){
			Gint += ga/rho;
			if( Gint <= 0 ) tauW = 2.2204e-16; 
			else tauW = Stau / ( log( (Gint+1) / Gint) );
			gW = exp(-double(t)/tauW)-tauW/sqrt(double((t)*N));
			
			
			if( gW < 0 ) {               // this W is taken as optimal
			  Wopt=t; 
			  tmax = min(tmax,2*t); 
			  flag = false; // Gamma up to tmax
			}
		}
		
		t++;
	}
	 
	 
	if(flag){
		cout << "WARNING: windowing condition failed up to W = " << tmax << endl;
  		Wopt=tmax;
		cout << "N: " << N  << "\tGint: " << Gint << "\ttauw: " << tauW << "\tgW: " << gW << "\tWopt: " << Wopt << "\ttmax: " << tmax << endl;
  		return;
	}

	cout << "N: " << N  << "\tGint: " << Gint << "\ttauw: " << tauW << "\tgW: " << gW << "\tWopt: " << Wopt << "\ttmax: " << tmax << endl;
	 
    double CFbbopt = GammaFbb[0];
    for(int i = 1; i <= Wopt; i++)
    	 CFbbopt += 2. * GammaFbb[i];   // first estimate
	cout << "CFbbopt: " << CFbbopt << "\t(first estimate)"  << endl;
	 
	 
	for(int i = 0; i < GammaFbb.size(); ++i)
		GammaFbb[i] += CFbbopt / double(N); // bias in Gamma corrected

    CFbbopt = GammaFbb[0];
    for(int i = 1; i <= Wopt; i++)
    	 CFbbopt += 2. * GammaFbb[i];   // refined estimate
	cout << "CFbbopt: " << CFbbopt << "\t(refined estimate)" << endl;
	 
	double sigmaF = sqrt(CFbbopt/double(N)); // error of F
	vector<double> norm_rho;
	for(int i = 0; i < GammaFbb.size(); ++i)
		norm_rho.push_back(GammaFbb[i]/GammaFbb[0]);  // normalized autocorr.
		
		
	vector<double> tauintFbb;
	double csum = -0.5;
	for(int i = 0; i < norm_rho.size(); ++i){
		csum += norm_rho[i];
		tauintFbb.push_back(csum);
		//cout << i << '\t' << corr[i] << '\t' << GammaFbb[i]  << '\t' << norm_rho[i] << '\t' << tauintFbb[i] << endl;
	}
	
	
	double value   = avg;
	double dvalue  = sigmaF;
	double ddvalue = dvalue*sqrt((double(Wopt)+0.5)/double(N));
	double tauint  = tauintFbb[Wopt];
	double dtauint = tauint*2.*sqrt((double(Wopt)-tauint+0.5)/double(N));
	
	cout << "value: " << avg << endl;
	cout << "dvalue: " << dvalue << endl;
	cout << "ddvalue: " << ddvalue << endl;
	cout << "tauint: " << tauint << endl;
	cout << "dtauint: " << dtauint << endl;
	cout << "Wopt: " << Wopt << endl;
	
	//a0.stop();
	//std::cout << "AC Time: " << a0.getElapsedTime() << " s" << std::endl;
	//std::cout << "-----------------------------------------" << std::endl;
}











































void calculateCorTime1(int miniter, int iter, std::vector<double> &gamma, int &nsweep){
	//Timer a0; a0.start();
	static bool calc =  false;
	if(gamma.size() < miniter) return;
	if(calc) return;
	
	int N = gamma.size() / 2;
	
	double avg = 0.;
	for(int i = 0; i < N; ++i) avg += gamma[i];
	avg /= double(N);

	/*double rho=0.;
	for(int i = 0; i < N; ++i){
		double tmp = gamma[i] - avg;
		rho += tmp * tmp;
	}
	rho /= double(N);*/

	std::vector<double> corr;
	double rho=0.;
	for(int j = 0; j < N; ++j){
		double g0 = 0.;
		double gt = 0.;
		for(int i = 0; i < N-j; ++i){
			g0 += gamma[i];
			gt += gamma[i+j];
		}
		double norm = 1.0 / double(N-j);
		g0 *= norm;
		gt /= norm;
		double ga = 0.;
		for(int i = 0; i < N-j; ++i){
			ga += (gamma[i] - g0) * (gamma[i+j] - gt);
			//ga += (gamma[i] - avg) * (gamma[i+j] - avg);
		}
		ga *= norm;
		if(j==0) rho = ga;//std::cout << rho << '\t' << ga << '\t' << ga/rho << std::endl;
		corr.push_back(ga/rho);
	}
	

	double tau_int = 0.5;
	for(int i = 1; i < N;++i){
		tau_int += corr[i-1];
		int tauInt = int(4.*tau_int+1);
		if( i > tauInt ) {
			nsweep = i;
			std::cout << "iter: " << iter << "\tN: " << N << "\tsweeps: " << nsweep << "\ttau_int: " << tau_int << "\tint(4.*tau_int+1): " << tauInt << std::endl;
			//gamma.clear();
			//calc = true;
			break;
		}
	}	
	//a0.stop();
	//std::cout << "AC Time: " << a0.getElapsedTime() << " s" << std::endl;
	//std::cout << "-----------------------------------------" << std::endl;
}



void calculateCorTime11111(int miniter, int iter, std::vector<double> &gamma, int &nsweep){
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
			std::cout << "iter1: " << iter << "\tsweeps: " << nsweep << "\ttau_int: " << tau_int << "\tint(4.*tau_int+1): " << tauInt << std::endl;
			gamma.clear();
			//calc = true;
			break;
		}
	}	
	//a0.stop();
	//std::cout << "AC Time: " << a0.getElapsedTime() << " s" << std::endl;
	//std::cout << "-----------------------------------------" << std::endl;
}


}



