//
// Created by fabian on 9/4/24.
//

#ifndef SUBDWARF_RV_SIMULATION_MATHS_H
#define SUBDWARF_RV_SIMULATION_MATHS_H

#endif //SUBDWARF_RV_SIMULATION_MATHS_H

#include "vector"

using namespace std;

vector<double> randomValuesFromBins(const vector<double>& binCenters, const vector<double>& weights, size_t N, bool use_seed = false, unsigned int seed = 0, bool log_spaced=true, bool add_randomness=true);
vector<double> sinusoid_wrapper(vector<double> x, double amplitude, double period, double offset, double phase, double Noise=0.0);
double randomValueInRange(double lo, double hi, bool use_seed=false, unsigned seed=0);
vector<double> genOptimalPeriodogramSamples(const vector<double>& t, double sample_factor = 10, double forced_min_p=0.0, double forced_max_p=0.0, int forced_N=0);
vector<double> binCenterFromEdges(vector<double> binEdges, bool islog=false);
double vrad_pvalue(const vector<double>& vrad, const vector<double>& vrad_err);
tuple<vector<double>, vector<double>, vector<double>, vector<double>> sinusoidMonteCarlo(const vector<double>& x, const vector<double>& y, const vector<double>& y_err, double period, int N_sim, double amp_0 = 100, double phase_0=.5, double offset_0=0.0, double amp_step = 10., double phase_step = 0.1, double offset_step = 10., int N_burn_in = 1000);