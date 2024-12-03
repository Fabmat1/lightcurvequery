//
// Created by fabian on 9/1/24.
//

#include <vector>
#include "vector_operations.h"
#include "exceptions.h"
#include <cmath>
#include <numeric>
#include <algorithm> // for transform
#include <stdexcept>
#include <cstring>
#include <complex>
#include <iostream>
#include <fftw3.h>


vector<double> extirpolate(vector<double> x, vector<double> y, int nfft, int mfft);

vector<double> arange(int m);

double vprod(vector<double> vector1);

vector<double> vround(vector<double> vec);

vector<double> vfloor(vector<double> vec);

using namespace std;

vector<double> linspace(double start, double end, int num) {
    vector<double> result(num);

    if (num == 0) {
        return result; // Return an empty vector if num is 0
    }
    if (num == 1) {
        result[0] = start;
        return result; // Return a vector with the start value if num is 1
    }

    double step = (end - start) / (num - 1);

    for (int i = 0; i < num; ++i) {
        result[i] = start + i * step;
    }

    return result;
}

vector<double> power(vector<double> vec, double p){
    vector<double> out(vec.size());
    for (int i = 0; i < vec.size(); ++i) {
        out[i] = pow(vec[i], p);
    }
    return out;
}

double vsum(vector<double> vec){
    double out = 0;
    for (auto& n : vec)
        out += n;
    return out;
}


double vdot(vector<double> vec1, vector<double> vec2){
    return inner_product(begin(vec1), end(vec1), begin(vec2), 0.0);
}


vector<double> vmult(vector<double> vec, double a){
    vector<double> result(vec.size());
    transform(vec.begin(), vec.end(), result.begin(),
              [a](double x) { return x * a; });
    return result;
};

vector<complex<double>> vcmult(vector<double> vec, complex<double> a){
    vector<complex<double>> vec_c (vec.size());
    transform(vec.begin(), vec.end(), vec_c.begin(), []( double da ) {
        return complex<double>( da, 0.0 );});

    vector<complex<double>> result(vec_c.size());
    transform(vec.begin(), vec.end(), result.begin(),
              [a](double x) { return x * a;});

    return result;
};


vector<double> vvdivide(vector<double> vec1, vector<double> vec2){
    vector<double> result(vec1.size());
    transform(vec1.begin(), vec1.end(), vec2.begin(), result.begin(),
              [](double a, double b) { return b != 0.0 ? a / b : 0.0; });
    return result;
}

vector<double> vdivide(vector<double> vec1, const double a){
    vector<double> result(vec1.size());
    transform(vec1.begin(), vec1.end(), result.begin(),
              [&a](double x) { return a != 0.0 ? x / a : 0.0; });
    return result;
}


vector<double> invert(vector<double> vec){
    vector<double> out(vec.size());

    if (vec.size() >= 1000){
        #pragma omp parallel for
        for (int i = 0; i < vec.size(); ++i) {
            out[i] = 1/vec[i];
        }
    }
    else {
        for (int i = 0; i < vec.size(); ++i) {
            out[i] = 1/vec[i];
        }
    }

    return out;
}


pair<vector<double>, vector<double>> broadcast_and_flatten(vector<double> a, vector<double> b) {
    if (a.size() != b.size() && a.size() != 1 && b.size() != 1) {
        throw runtime_error("Cannot broadcast arrays of different sizes");
    }

    if (a.size() < b.size() && a.size() == 1) {
        a.resize(b.size(), a[0]);
    } else if (b.size() < a.size() && b.size() == 1) {
        b.resize(a.size(), b[0]);
    }

    return {a, b};
}



unsigned long long bitceil(unsigned long long N) {
    // Find the bit (i.e. power of 2) immediately greater than or equal to N
    // Note: this works for numbers up to 2 ** 64.
    if (N == 0)
        return 1;
    else
        return 1 << (unsigned long long)(log2(N - 1) + 1);
}


vector<double> vadd(vector<double> vec, double a){
    vector<double> result(vec.size());
    transform(vec.begin(), vec.end(), result.begin(),
              [a](double x) { return x + a; });
    return result;
}

vector<double> vlog(vector<double> vec){
    vector<double> result(vec.size());
    transform(vec.begin(), vec.end(), result.begin(),
              [](double x) { return log(x); });
    return result;
}


vector<double> vmod(vector<double> vec, double a){
    vector<double> result(vec.size());
    transform(vec.begin(), vec.end(), result.begin(),
              [a](double x) { return fmod(x, a); });
    return result;
}

vector<double> vclip(vector<double> vec, double low, double high){
    vector<double> result(vec.size());
    transform(vec.begin(), vec.end(), result.begin(),
              [low, high](double x) {
                          if(x <= high && x >= low) {return x;}
                          else if (x > high) {return high;}
                          else {return low;}});
    return result;
}


vector<double> vvadd(vector<double> vec1, vector<double> vec2){
    vector<double> result(vec1.size());
    transform(vec1.begin(), vec1.end(), vec2.begin(), result.begin(),
              [](double a, double b) { return  a + b;});
    return result;
}


vector<double> vvmult(vector<double> vec1, vector<double> vec2){
    vector<double> result(vec1.size());
    transform(vec1.begin(), vec1.end(), vec2.begin(), result.begin(),
              [](double a, double b) { return  a * b;});
    return result;
}


vector<complex<double>> compute_ifft(const vector<double>& grid, size_t N) {
    size_t M = grid.size();
    vector<complex<double>> out(M);

    // FFTW plan
    fftw_plan plan = fftw_plan_dft_r2c_1d(M, const_cast<double*>(grid.data()), reinterpret_cast<fftw_complex*>(out.data()), FFTW_ESTIMATE);

    // Execute the plan
    fftw_execute(plan);

    // Normalize and select first N elements
    vector<complex<double>> result(N);
    for (size_t i = 0; i < N; ++i) {
        result[i] = out[i] / static_cast<double>(M);  // Normalize by dividing by the size of input
        result[i].imag(-result[i].imag());
    }

    // Clean up FFTW plan
    fftw_destroy_plan(plan);

    return result;
}


vector<complex<double>> compute_ifft_complex(const vector<complex<double>>& grid, size_t N) {
    size_t M = grid.size();
    vector<complex<double>> out(M);

    // FFTW plan
    fftw_plan plan = fftw_plan_dft_1d(M,
                                      reinterpret_cast<fftw_complex*>(const_cast<complex<double>*>(grid.data())),
                                      reinterpret_cast<fftw_complex*>(out.data()),
                                      FFTW_BACKWARD, FFTW_ESTIMATE);

    // Execute the plan
    fftw_execute(plan);

    // Normalize and select first N elements
    vector<complex<double>> result(N);
    for (size_t i = 0; i < N; ++i) {
        result[i] = out[i] / static_cast<double>(M);  // Normalize by dividing by the size of input
    }

    // Clean up FFTW plan
    fftw_destroy_plan(plan);

    return result;
}


double minimum_diff(const vector<double>& vec) {
    if (vec.size() < 2) {
        throw invalid_argument("Vector must contain at least two elements");
    }

    // Lambda to compute difference between consecutive elements
    auto diff = [&vec](size_t i) {
        return vec[i + 1] - vec[i];
    };

    // Find the minimum difference using min_element
    auto min_diff = min_element(vec.begin(), vec.end() - 1,
                                     [&diff, &vec](double a, double b) {
                                         return diff(&a - vec.data()) < diff(&b - vec.data());
                                     });

    // Return the minimum difference
    return diff(min_diff - vec.begin());
}


double maximum_diff(const vector<double>& vec) {
    if (vec.size() < 2) {
        throw invalid_argument("Vector must contain at least two elements");
    }

    // Lambda to compute difference between consecutive elements
    auto diff = [&vec](size_t i) {
        return vec[i + 1] - vec[i];
    };

    // Find the minimum difference using min_element
    auto max_diff = max_element(vec.begin(), vec.end() - 1,
                                [&diff, &vec](double a, double b) {
                                    return diff(&a - vec.data()) < diff(&b - vec.data());
                                });

    // Return the minimum difference
    return diff(max_diff - vec.begin());
}


double ptp(const vector<double>& vec){
    double v_min = *min_element(begin(vec), end(vec));
    double v_max = *max_element(begin(vec), end(vec));
    return v_max-v_min;
}


double get_min(const vector<double>& vec) {
    if (vec.empty()) {
        throw invalid_argument("Vector is empty");
    }
    return *min_element(vec.begin(), vec.end());
}


double get_max(const vector<double>& vec) {
    if (vec.empty()) {
        throw invalid_argument("Vector is empty");
    }
    return *max_element(vec.begin(), vec.end());
}


pair<vector<double>, vector<double>> trig_sum(vector<double> t, vector<double> h, double df, int N,
                                               double f0, double freq_factor, int oversampling, int Mfft){

    df *= freq_factor;
    f0 *= freq_factor;


    if (df <= 0){ throw ValueError("df must be positive");};
    pair<vector<double>, vector<double>> p = broadcast_and_flatten(t, h);
    t = p.first;
    h = p.second;

    if (Mfft <= 0)
    {
        throw ValueError("Mfft must be positive");
    }

    // required size of fft is the power of 2 above the oversampling rate


    unsigned Nfft_temp = bitceil(N * oversampling);
    int Nfft;
    memcpy(&Nfft, &Nfft_temp, sizeof(int));
    double t0 = *min_element(t.begin(), t.end());
    vector<complex<double>> h_complex(h.size());

    if (f0 > 0) {
        vector<complex<double>> exp_exponent = (vcmult(vadd(t, -t0), 2j * M_PI * f0));
        vector<complex<double>> exp_vec(exp_exponent.size());

        for (int i = 0; i < exp_exponent.size(); ++i) {
            h_complex[i] = h[i]*exp(exp_exponent[i]);;
        }
        vector<double> tnorm = vmod((vmult(vadd(t, -t0), Nfft * df)), Nfft);
        vector<complex<double>> grid = extirpolate_complex(tnorm, h_complex, Nfft, Mfft);
        vector<complex<double>> fftgrid = compute_ifft_complex(grid, N);
        vector<double> C(fftgrid.size());
        vector<double> S(fftgrid.size());
        if (t0 != 0) {
            vector<double> f = vadd(vmult(arange(N), df), f0);
            vector<complex<double>> exp_exponent = vcmult(f, 2j * M_PI * t0);
            for (int i = 0; i < fftgrid.size(); ++i) {
                fftgrid[i] *= exp(exp_exponent[i]);
            }
        }
        for (int i = 0; i < fftgrid.size(); ++i) {
            C[i] = fftgrid[i].real();
            S[i] = fftgrid[i].imag();
            C[i] *= Nfft;
            S[i] *= Nfft;
        }
        return make_pair(S, C);
    }
    else{
        vector<double> tnorm = vmod((vmult(vadd(t, -t0), Nfft * df)), Nfft);
        vector<double> grid = extirpolate(tnorm, h, Nfft, Mfft);
        vector<complex<double>> fftgrid = compute_ifft(grid, N);
        vector<double> C(fftgrid.size());
        vector<double> S(fftgrid.size());
        if (t0 != 0) {
            vector<double> f = vadd(vmult(arange(N), df), f0);
            vector<complex<double>> exp_exponent = vcmult(f, 2j * M_PI * t0);
            for (int i = 0; i < fftgrid.size(); ++i) {
                fftgrid[i] *= exp(exp_exponent[i]);
            }
        }
        for (int i = 0; i < fftgrid.size(); ++i) {
            C[i] = fftgrid[i].real();
            S[i] = fftgrid[i].imag();
            C[i] *= Nfft;
            S[i] *= Nfft;
        }
        return make_pair(S, C);
    }


}


void removeIntegerValues(vector<double>& x, vector<double>& y) {
    // Start from the end of the vector to avoid invalidating indices when erasing elements.
    for (int i = x.size() - 1; i >= 0; --i) {
        if (fmod(x[i], 1.0) == 0.0) {
            // Erase elements at index i in both x and y
            x.erase(x.begin() + i);
            y.erase(y.begin() + i);
        }
    }
}


double pyint(double a){
    if (a >= 0){
        return floor(a);
    }
    else{
        return ceil(a);
    }
}


vector<double> extirpolate(vector<double> x, vector<double> y, int N, int M) {
    pair<vector<double>, vector<double>> p = broadcast_and_flatten(x, y);
    if (N==0){
        N = (int)round(*max_element(x.begin(), x.end()) + 0.5 * M + 1);
    }
    x = p.first;
    y = p.second;

    vector<double> result(N, 0);


    for (int i = 0; i < x.size(); ++i) {
        if (fmod(x[i], 1) == 0){
            int ind = (int)x[i];
            result[ind] += y[i];
        }
    }

    removeIntegerValues(x, y);

    vector<double> redarr = vfloor(vadd(x, -floor(M / 2.0)));
    vector<double> ilo = vclip(redarr, 0, N-M);
    ilo = vmult(ilo, -1);

    vector<double> M_arange = arange(M);
    vector<vector<double>> numerator_matrix (M,vector<double>(x.size()));
    for (int i = 0; i < M; ++i) {
        numerator_matrix[i] = vadd(vvadd(x, ilo), -M_arange[i]);
    }
    vector<double> numerator (y.size(), 1);
    for (int i = 0; i < numerator.size(); ++i) {
        for (int j = 0; j < M; ++j) {
            numerator[i] *= numerator_matrix[j][i];
        }
        numerator[i] *= y[i];
    }

    double denominator = tgamma(M);
    ilo = vmult(ilo, -1);

    for (int j = 0; j < M; j++) {
        if (j > 0) {
            denominator *= (double)j / (j - M);
        }
        vector<double> ind = vadd(ilo, (M - 1 - j));
        for (int i = 0; i < ind.size(); ++i) {
            int index = (int)round(ind[i]);
            result[index] += numerator[i] / (denominator * (x[i]-index));
            result[index] = pyint(result[index]);
        }
    }
    return result;
}


void removeIntegerValues_c(vector<double>& x, vector<complex<double>>& y) {
    // Start from the end of the vector to avoid invalidating indices when erasing elements.
    for (int i = x.size() - 1; i >= 0; --i) {
        if (fmod(x[i], 1.0) == 0.0) {
            // Erase elements at index i in both x and y
            x.erase(x.begin() + i);
            y.erase(y.begin() + i);
        }
    }
}


vector<complex<double>> extirpolate_complex(vector<double> x, vector<complex<double>> y, int N, int M) {
    if (N==0){
        N = (int)round(*max_element(x.begin(), x.end()) + 0.5 * M + 1);
    }

    vector<complex<double>> result(N, 0);


    for (int i = 0; i < x.size(); ++i) {
        if (fmod(x[i], 1) == 0){
            int ind = (int)x[i];
            result[ind] += y[i];
        }
    }

    removeIntegerValues_c(x, y);

    vector<double> redarr = vfloor(vadd(x, -floor(M / 2.0)));
    vector<double> ilo = vclip(redarr, 0, N-M);
    ilo = vmult(ilo, -1);

    vector<double> M_arange = arange(M);
    vector<vector<double>> numerator_matrix (M,vector<double>(x.size()));
    for (int i = 0; i < M; ++i) {
        numerator_matrix[i] = vadd(vvadd(x, ilo), -M_arange[i]);
    }
    vector<complex<double>> numerator (y.size(), 1);
    for (int i = 0; i < numerator.size(); ++i) {
        for (int j = 0; j < M; ++j) {
            numerator[i] *= numerator_matrix[j][i];
        }
        numerator[i] *= y[i];
    }

    double denominator = tgamma(M);
    ilo = vmult(ilo, -1);

    for (int j = 0; j < M; j++) {
        if (j > 0) {
            denominator *= (double)j / (j - M);
        }
        vector<double> ind = vadd(ilo, (M - 1 - j));
        for (int i = 0; i < ind.size(); ++i) {
            int index = (int)round(ind[i]);
            result[index] += numerator[i] / (denominator * (x[i]-index));
        }
    }
    return result;
}




vector<double> vfloor(vector<double> vec) {
    vector<double> result(vec.size());
    transform(vec.begin(), vec.end(), result.begin(),
              [](double a) { return  floor(a);});
    return result;
}

vector<double> vround(vector<double> vec) {
    vector<double> result(vec.size());
    transform(vec.begin(), vec.end(), result.begin(),
              [](double a) { return  round(a);});
    return result;
}

double vprod(vector<double> vec) {
    double result = 1.;
    for (auto i : vec) {
        result *= i;
    }
    return result;
}

vector<double> arange(int m) {
    vector<double> result(m); // Declare vector with size m
    for (int i = 0; i < m; ++i) {
        result[i] = static_cast<double>(i); // Directly assign values
    }
    return result;
}


