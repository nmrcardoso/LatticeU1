#include <cstdlib>
#include <cstdio>
#include <string>
#include <cstring>
#include <map>
#include <iostream>
#include <vector>
#include <math.h>


#include <cuda_vector_types.h>


namespace U1{

using namespace std;





double Sum(const vector<double> *data){
    double s = 0.0;
    for (auto it = data->begin(); it != data->end(); it++) 
        s = s + *it;
    return s;
}
double Sum(const vector<double> *data, double a){
    double s = 0.0;
    for (auto it = data->begin(); it != data->end(); it++) 
        s += *it -a;
    return s;
}

double Mean(const vector<double> *data){
    return Sum(data)/double(data->size());
}





double Sum(const vector<double> &data){
    double s = 0.0;
    for (int i = 0; i < data.size(); ++i)
        s += data[i];
    return s;
}
double Sum(const vector<double> &data, double a){
    double s = 0.0;
    for (int i = 0; i < data.size(); ++i)
        s += data[i] - a;
    return s;
}
double SquareSum(const vector<double> &data, double a){
    double s = 0.0;
    for (int i = 0; i < data.size(); ++i){
    	double sq = data[i] - a;
        s += sq * sq;
    }
    return s;
}




double Mean(const vector<double> &data){
    return Sum(data)/double(data.size());
}





double2 jackknife(const vector<double> *data, double (*f)(double)){
	int length = data->size();
	double mean = Sum(data);
	vector<double> trials;
    for (auto it = data->begin(); it != data->end(); it++) {
		double tmp = (mean - *it)/double(length-1);
		trials.push_back(f(tmp));
	}
	mean /= double(length);
	mean = f(mean);
	
	double jmean = Mean(trials);
	double err0 = SquareSum(trials, jmean);
	double norm = double(length-1) / double(length);
	return make_double2(mean, sqrt(err0 * norm));
}


double2 jackknife(const vector<double> &data, double (*f)(double)){
	int length = data.size();
	double mean = Sum(data);
	vector<double> trials;
	for(int i = 0; i < length; ++i){
		double tmp = (mean - data[i])/double(length-1);
		trials.push_back(f(tmp));
	}
	mean /= double(length);
	mean = f(mean);
	
	double jmean = Mean(trials);
	double err0 = SquareSum(trials, jmean);
	double norm = double(length-1) / double(length);
	return make_double2(mean, sqrt(err0 * norm));
}




double2 jackknife(const vector<double> &data){
	int length = data.size();
	double mean = 0.0;
	for(int i = 0; i < length; ++i) mean += data[i];
	vector<double> trials;
	for(int i = 0; i < length; ++i) trials.push_back((mean - data[i])/double(length-1));
	mean /= double(length);
		
	double jmean = Mean(trials);
	double err0 = SquareSum(trials, jmean);
	double norm = double(length-1) / double(length);
	return make_double2(mean, sqrt(err0 * norm));
}



double2 susceptibility (const vector<double> &data){
    double mu = Mean(data);
    int length = data.size();
    vector<double> jack;
    for (int i = 0; i < length; ++i){
    	double tmp = (data[i] - mu)*(data[i] - mu);
        jack.push_back(tmp);
    }
    return jackknife(jack);
}

}



