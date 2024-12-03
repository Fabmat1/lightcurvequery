// Sourced from https://github.com/exord/gls and adapted

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


vector<double> GLS(const vector<double>& prange, const vector<double>& x, const vector<double>& y, const vector<double>& y_err) {
    double omega;
    double ccos, ssin;
    double Y, C, S;
    double YYhat, YChat, YShat, CChat, SShat, CShat;
    double YY, YC, YS, CC, SS, CS;
    double D;

    vector<double> powers(prange.size());

    Y = 0;
    YYhat = 0;

    #pragma omp parallel for
    for (int ii = 0; ii < x.size(); ++ii)
    {
        Y += y_err[ii] * y[ii];
        YYhat += y_err[ii] * y[ii] * y[ii];
    }

    #pragma omp parallel for
    for (int jj = 0; jj < prange.size(); ++jj)

    {
        omega = 2.0 * M_PI  / prange[jj];

        C = S = 0.0;
        YChat = YShat = CChat = SShat = CShat = 0.0;

        for (int ii = 0; ii < x.size(); ++ii)
        {
            ccos = cos(omega * x[ii]);
            ssin = sin(omega * x[ii]);

            C += y_err[ii] * ccos;
            S += y_err[ii] * ssin;

            YChat += y_err[ii] * y[ii] * ccos;
            YShat += y_err[ii] * y[ii] * ssin;
            CChat += y_err[ii] * ccos * ccos;
            SShat += y_err[ii] * ssin * ssin;
            CShat += y_err[ii] * ccos * ssin;
        }

        YY = YYhat - Y*Y;
        YC = YChat - Y*C;
        YS = YShat - Y*S;
        CC = CChat - C*C;
        SS = SShat - S*S;
        CS = CShat - C*S;

        D = CC*SS - CS*CS;

        powers[jj] = (SS * YC * YC + CC * YS * YS - 2.0 * CS * YC * YS) / (YY * D);
    }
    return powers;
}

template <typename T> int sgn(T val)
{
    return (T(0) < val) - (val < T(0));
}

vector<double> solveLinearSystem(const vector<vector<double>>& XTX, const vector<double>& XTy) {
    int n = XTy.size();

    // Convert vector to Eigen::MatrixXd
    Eigen::MatrixXd XTX_Eigen(n, n);
    Eigen::VectorXd XTy_Eigen(n);

    for (int i = 0; i < n; ++i) {
        XTy_Eigen(i) = XTy[i];
        for (int j = 0; j < n; ++j) {
            XTX_Eigen(i, j) = XTX[i][j];
        }
    }

    // Solve the system XTX * beta = XTy
    Eigen::VectorXd beta = XTX_Eigen.colPivHouseholderQr().solve(XTy_Eigen);

    // Convert Eigen::VectorXd to vector<double>
    vector<double> result(n);
    for (int i = 0; i < n; ++i) {
        result[i] = beta(i);
    }

    return result;
}

//Ripped from astropy and translated to C++ cuz its faaaster
vector<double> gls_fast (vector<double>t,
                         vector<double>y,
                         vector<double>dy,
                         double f0,
                         double df,
                         int Nf,
                         string normalization,
                         bool fit_mean,
                         bool center_data,
                         int nterms,
                         bool use_fft){

    if (nterms == 0 && !fit_mean)
    {throw ValueError("Cannot have nterms = 0 without fitting bias");};

    if (f0 < 0) {
        throw ValueError("Frequencies must be positive");
    }
    if (df <= 0) {
        throw ValueError("Frequency steps must be positive");
    }
    if (Nf <= 0) {
        throw ValueError("Number of frequencies must be positive");
    }

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
        return 0.5 * (sgn(m - n) * Sw[abs(m - n)][i] + Sw[m + n][i]);
    };
    funcs["CS"] = [&Sw](int m, int n, int i) -> double {
        return 0.5 * (sgn(n - m) * Sw[abs(n - m)][i] + Sw[n + m][i]);
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

    if (normalization == "psd") { p = vmult(p, 0.5); }
    else if (normalization == "standard") { p = vmult(p, 1/chi2_ref) ;}
    else if (normalization == "log") { p = vmult(vlog(vadd(vmult(p, -1/chi2_ref), 1)), -1) ;}
    else if (normalization == "model") { vvmult(p, invert(vadd(vmult(p, -1), chi2_ref))); }
    else {throw ValueError("normalization not recognized");}

    return p;
};