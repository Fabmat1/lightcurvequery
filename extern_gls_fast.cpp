//
// Created by fabian on 9/25/24.
//

#include "lomb_scargle_periodogram.h"
#include "vector_operations.h"
#include "exceptions.h"
#include <iostream>
#include <cmath>
#include <omp.h>
#include <vector>
#include <map>
#include <functional>
#include <Eigen/Dense>

using namespace std;

vector<double> doublePtrToVector(double* arr, size_t size) {
    return vector<double>(arr, arr + size);
}

double* vectorToDoublePtr(const vector<double>& vec) {
    return const_cast<double*>(vec.data());  // Remove const if necessary
}

template <typename T> int sign(T val)
{
    return (T(0) < val) - (val < T(0));
}

extern "C"{

//Ripped from astropy and translated to C++ cuz its faaaster
double* gls_fast_extern (double*t_pointer,
                         size_t t_size,
                         double*y_pointer,
                         size_t y_size,
                         double*dy_pointer,
                         size_t dy_size,
                         double f0,
                         double df,
                         int Nf,
                         int normalization,
                         bool fit_mean,
                         bool center_data,
                         int nterms            ){

    vector<double> t = doublePtrToVector(t_pointer, t_size);
    vector<double> y = doublePtrToVector(y_pointer, y_size);
    vector<double> dy = doublePtrToVector(dy_pointer, dy_size);

    vector<double> w = power(dy, -2);
    double ws = vsum(w);

    if (center_data or fit_mean){
        double dot_prdct = vdot(w, y);
        for (int i = 0; i < y.size(); ++i) {
            y[i] -= dot_prdct / ws;
        }
    }

    vector<double> yw = vvdivide(y, dy);
    double chi2_ref = vdot(yw, yw);

    // Here we build-up the matrices XTX and XTy using pre-computed
    // sums. The relevant identities are
    // 2 sin(mx) sin(nx) = cos(m-n)x - cos(m+n)x
    // 2 cos(mx) cos(nx) = cos(m-n)x + cos(m+n)x
    // 2 sin(mx) cos(nx) = sin(m-n)x + sin(m+n)x

    double yws = vsum(vvmult(y, w));

    vector<vector<double>> Sw(2 * nterms + 1,vector<double>(Nf, 0));
    vector<vector<double>> Cw(2 * nterms + 1,vector<double>(Nf, 0));

    vector<double> testvec = vvmult(y, w);
    for (int i = 0; i < Nf; ++i) {
        Cw[0][i] += ws;
    }
    pair<vector<double>, vector<double>> this_tsum;
    for (int i = 1; i < 2 * nterms + 1; ++i) {
        this_tsum = trig_sum(t, w, df, Nf, f0, (double)i);
        for (int j = 0; j < Nf; ++j) {
            Sw[i][j] = this_tsum.first[j];
            Cw[i][j] = this_tsum.second[j];
        }
    }

    vector<vector<double>> Syw(nterms + 1,vector<double>(Nf, 0));
    vector<vector<double>> Cyw(nterms + 1,vector<double>(Nf, 0));

    for (int i = 0; i < Nf; ++i) {
        Cyw[1][i] += yws;
    }
    for (int i = 1; i < nterms + 1; ++i) {
        this_tsum = trig_sum(t, vvmult(y, w), df, Nf, f0, (double)i);
        for (int j = 0; j < Nf; ++j) {
            Syw[i][j] =this_tsum.first[j];
            Cyw[i][j] =this_tsum.second[j];
        }
    }

    //# Now create an indexing scheme so we can quickly;
    //# build-up matrices at each frequency;
    vector<pair<string, int>> order(2*nterms+1);
    for (int i = 1; i < 2*nterms+1; i += 2) {
        order[i] = make_pair("S",(int)floor((i+1)/2.));
    }
    for (int i = 2; i < 2*nterms+1; i += 2) {
        order[i] = make_pair("C",(int)floor((i+1)/2.));
    }
    if (fit_mean){
        order[0] = make_pair("C", 0);
    }
    else{
        order.erase(order.begin());
    }

    map<string, function<double(int, int, int)>> funcs;
    // Populate the map with lambda functions
    funcs["S"] = [&Syw](int m, int i, int l) -> double { return Syw[m][i]; };
    funcs["C"] = [&Cyw](int m, int i, int l) -> double { return Cyw[m][i]; };
    funcs["SS"] = [&Cw](int m, int n, int i) -> double {
        return 0.5 * (Cw[abs(m - n)][i] - Cw[m + n][i]);
    };
    funcs["CC"] = [&Cw](int m, int n, int i) -> double {
        return 0.5 * (Cw[abs(m - n)][i] + Cw[m + n][i]);
    };
    funcs["SC"] = [&Sw](int m, int n, int i) -> double {
        return 0.5 * (sign(m - n) * Sw[abs(m - n)][i] + Sw[m + n][i]);
    };
    funcs["CS"] = [&Sw](int m, int n, int i) -> double {
        return 0.5 * (sign(n - m) * Sw[abs(n - m)][i] + Sw[n + m][i]);
    };

    auto compute_power = [&order, &funcs](int i) -> double {
        vector<vector<double>> XTX(order.size(), vector<double>(order.size()));
        pair<string, int> A;
        pair<string, int> B;
        for (int b = 0; b < order.size(); b++) {
            for(int a = 0; a < order.size(); a++){
                A = order[a];
                B = order[b];
                string funcstr = A.first + B.first;
                XTX[b][a] = funcs[funcstr](A.second, B.second, i);
            }
        }
        vector<double> XTy(order.size());
        for(int a = 0; a < order.size(); a++){
            A = order[a];
            XTy[a] = funcs[A.first](A.second, i, 0);
        }

        return vdot(XTy, solveLinearSystem(XTX, XTy));
    };
    vector<double> p(Nf);

    #pragma omp parallel for
    for (int i = 0; i < Nf; ++i) {
        p[i] = compute_power(i);
    }

    if (normalization == 0) { p = vmult(p, 0.5); }
    else if (normalization == 1) { p = vmult(p, 1/chi2_ref) ;}
    else if (normalization == 2) { p = vmult(vlog(vadd(vmult(p, -1/chi2_ref), 1)), -1) ;}
    else if (normalization == 3) { vvmult(p, invert(vadd(vmult(p, -1), chi2_ref))); }

    auto* data = new double[p.size()];

    for (int i = 0; i < p.size(); ++i) {
        data[i] = p[i];
    }

    return data;
};
}