//
// Created by fabian on 9/1/24.
//

#ifndef SUBDWARF_RV_SIMULATION_LOMB_SCARGLE_PERIODOGRAM_H
#define SUBDWARF_RV_SIMULATION_LOMB_SCARGLE_PERIODOGRAM_H

#endif //SUBDWARF_RV_SIMULATION_LOMB_SCARGLE_PERIODOGRAM_H

#include <vector>
#include <string>


using namespace std;

vector<double> GLS(const vector<double>& prange, const vector<double>& x, const vector<double>& y, const vector<double>& y_err);

vector<double> gls_fast (vector<double>t,
                         vector<double>y,
                         vector<double> dy,
                         double f0,
                         double df,
                         int Nf,
                         string normalization="standard",
                         bool fit_mean=true,
                         bool center_data=true,
                         int nterms=1,
                         bool use_fft=true);

vector<double> solveLinearSystem(const vector<vector<double>>& XTX, const vector<double>& XTy);