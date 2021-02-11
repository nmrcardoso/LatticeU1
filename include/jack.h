
#ifndef __JACK_H__
#define __JACK_H__


#include <vector>


namespace U1{

inline double potential(double in){
	return -log(in) / double(Grid(TDir()));
}


double Mean(const std::vector<double> *data);
double Mean(const std::vector<double> &data);

double Sum(const std::vector<double> *data);
double Sum(const std::vector<double> &data);

double2 jackknife(const std::vector<double> *data, double (*f)(double));

double2 jackknife(const std::vector<double> &data, double (*f)(double));

double2 jackknife(const std::vector<double> &data);

double2 susceptibility (const std::vector<double> &data);


}

#endif

