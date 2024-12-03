//
// Created by fabia on 08.09.2024.
//

#ifndef SUBDWARF_RV_SIMULATIONS_MODELS_H
#define SUBDWARF_RV_SIMULATIONS_MODELS_H

#endif //SUBDWARF_RV_SIMULATIONS_MODELS_H


#include "vector"
using namespace std;

struct Star {
    double amplitude;
    double period;
    double offset;
    double phase;
    int Npoints;

    vector<double> samples;
    vector<double> datapoints;
    vector<double> datapoint_errors;
    vector<double> periodogram_x;
    vector<double> periodogram_y;

    vector<vector<double>> period_amp_histogram;
    vector<vector<double>> period_phase_histogram;
    vector<vector<double>> period_offset_histogram;

    void calculate_orbit_prediciton(int Nx, int Ny, double amp_lim, double offset_lim);
    void process_star(int index, double forced_min_p=0.0, double forced_max_p=0.0, int forced_N=0);
    void sort_samples();
};
