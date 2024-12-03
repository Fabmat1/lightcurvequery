//
// Created by fabian on 9/1/24.
//

#ifndef SUBDWARF_RV_SIMULATION_VECTOR_OPERATIONS_H
#define SUBDWARF_RV_SIMULATION_VECTOR_OPERATIONS_H

#endif //SUBDWARF_RV_SIMULATION_VECTOR_OPERATIONS_H

#include <vector>
#include <complex>

using namespace std;

double vdot(vector<double> vec1, vector<double> vec2);
vector<double> linspace(double start, double end, int num);
vector<double> invert(vector<double> vec);
vector<double> power(vector<double> vec, double pow);
double vsum(vector<double> vec);
vector<double> vvdivide(vector<double> vec1, vector<double> vec2);
vector<double> vdivide(vector<double> vec1, double a);
vector<double> vmult(vector<double> vec1, double a);
vector<double> vadd(vector<double> vec1, double a);
vector<double> vlog(vector<double> vec);
double get_min(const vector<double>& vec);
double get_max(const vector<double>& vec);
double minimum_diff(const vector<double>& vec);
double maximum_diff(const vector<double>& vec);
double ptp(const vector<double>& vec);
vector<double> vvmult(vector<double> vec1, vector<double> vec2);
vector<double> extirpolate(vector<double> x, vector<double>y, int N, int M);
vector<complex<double>> extirpolate_complex(vector<double> x, vector<complex<double>>y, int N, int M);
pair<vector<double>, vector<double>> trig_sum(vector<double> t, vector<double> h, double df, int N,
                                               double f0=0.0, double freq_factor=1.0, int oversampling=5, int Mfft=4);