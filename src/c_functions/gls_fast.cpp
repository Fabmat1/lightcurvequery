//
// Self-contained Generalized Lomb-Scargle periodogram.
// Translated from astropy, consolidated into one file.
//
// Dependencies: Eigen3 (headers only), FFTW3, OpenMP
//

#include <cmath>
#include <cstring>
#include <complex>
#include <vector>
#include <algorithm>
#include <numeric>
#include <stdexcept>
#include <string>
#include <utility>

#include <omp.h>
#include <fftw3.h>
#include <Eigen/Dense>

using namespace std;

// =====================================================================
// Exception
// =====================================================================

class ValueError : public exception {
    string m_msg;
public:
    explicit ValueError(string msg) : m_msg(std::move(msg)) {}
    const char* what() const noexcept override { return m_msg.c_str(); }
};

// =====================================================================
// Scalar helpers
// =====================================================================

template <typename T>
static int sign(T val) {
    return (T(0) < val) - (val < T(0));
}

static double pyint(double a) {
    return (a >= 0) ? floor(a) : ceil(a);
}

static unsigned long long bitceil(unsigned long long N) {
    if (N == 0) return 1;
    return 1ULL << (unsigned long long)(log2((double)(N - 1)) + 1);
}

// =====================================================================
// Real vector operations
// =====================================================================

static vector<double> arange(int m) {
    vector<double> r(m);
    for (int i = 0; i < m; ++i) r[i] = static_cast<double>(i);
    return r;
}

static double vsum(const vector<double>& vec) {
    double s = 0;
    for (auto v : vec) s += v;
    return s;
}

static double vdot(const vector<double>& a, const vector<double>& b) {
    return inner_product(a.begin(), a.end(), b.begin(), 0.0);
}

static vector<double> power(const vector<double>& vec, double p) {
    vector<double> out(vec.size());
    for (size_t i = 0; i < vec.size(); ++i) out[i] = pow(vec[i], p);
    return out;
}

static vector<double> vmult(const vector<double>& vec, double a) {
    vector<double> r(vec.size());
    transform(vec.begin(), vec.end(), r.begin(),
              [a](double x) { return x * a; });
    return r;
}

static vector<double> vadd(const vector<double>& vec, double a) {
    vector<double> r(vec.size());
    transform(vec.begin(), vec.end(), r.begin(),
              [a](double x) { return x + a; });
    return r;
}

static vector<double> vvadd(const vector<double>& a, const vector<double>& b) {
    vector<double> r(a.size());
    transform(a.begin(), a.end(), b.begin(), r.begin(),
              [](double x, double y) { return x + y; });
    return r;
}

static vector<double> vvmult(const vector<double>& a, const vector<double>& b) {
    vector<double> r(a.size());
    transform(a.begin(), a.end(), b.begin(), r.begin(),
              [](double x, double y) { return x * y; });
    return r;
}

static vector<double> vvdivide(const vector<double>& a, const vector<double>& b) {
    vector<double> r(a.size());
    transform(a.begin(), a.end(), b.begin(), r.begin(),
              [](double x, double y) { return y != 0.0 ? x / y : 0.0; });
    return r;
}

static vector<double> vmod(const vector<double>& vec, double a) {
    vector<double> r(vec.size());
    transform(vec.begin(), vec.end(), r.begin(),
              [a](double x) { return fmod(x, a); });
    return r;
}

static vector<double> vclip(const vector<double>& vec, double lo, double hi) {
    vector<double> r(vec.size());
    transform(vec.begin(), vec.end(), r.begin(),
              [lo, hi](double x) {
                  if (x < lo) return lo;
                  if (x > hi) return hi;
                  return x;
              });
    return r;
}

static vector<double> vfloor(const vector<double>& vec) {
    vector<double> r(vec.size());
    transform(vec.begin(), vec.end(), r.begin(),
              [](double x) { return floor(x); });
    return r;
}

// =====================================================================
// Complex vector helpers
// =====================================================================

static vector<complex<double>> vcmult(const vector<double>& vec,
                                       complex<double> a) {
    vector<complex<double>> r(vec.size());
    transform(vec.begin(), vec.end(), r.begin(),
              [a](double x) { return x * a; });
    return r;
}

// =====================================================================
// Broadcast
// =====================================================================

static pair<vector<double>, vector<double>>
broadcast_and_flatten(vector<double> a, vector<double> b) {
    if (a.size() != b.size() && a.size() != 1 && b.size() != 1)
        throw runtime_error("Cannot broadcast arrays of different sizes");
    if (a.size() < b.size() && a.size() == 1) a.resize(b.size(), a[0]);
    else if (b.size() < a.size() && b.size() == 1) b.resize(a.size(), b[0]);
    return {a, b};
}

// =====================================================================
// FFT wrappers  (FFTW3)
// =====================================================================

static vector<complex<double>> compute_ifft(const vector<double>& grid,
                                             size_t N) {
    size_t M = grid.size();
    vector<complex<double>> out(M);

    fftw_plan plan = fftw_plan_dft_r2c_1d(
        (int)M, const_cast<double*>(grid.data()),
        reinterpret_cast<fftw_complex*>(out.data()), FFTW_ESTIMATE);
    fftw_execute(plan);
    fftw_destroy_plan(plan);

    vector<complex<double>> result(N);
    double inv = 1.0 / static_cast<double>(M);
    for (size_t i = 0; i < N; ++i) {
        result[i] = complex<double>(out[i].real() * inv,
                                    -out[i].imag() * inv);
    }
    return result;
}

static vector<complex<double>> compute_ifft_complex(
        const vector<complex<double>>& grid, size_t N) {
    size_t M = grid.size();
    vector<complex<double>> out(M);

    fftw_plan plan = fftw_plan_dft_1d(
        (int)M,
        reinterpret_cast<fftw_complex*>(
            const_cast<complex<double>*>(grid.data())),
        reinterpret_cast<fftw_complex*>(out.data()),
        FFTW_BACKWARD, FFTW_ESTIMATE);
    fftw_execute(plan);
    fftw_destroy_plan(plan);

    vector<complex<double>> result(N);
    double inv = 1.0 / static_cast<double>(M);
    for (size_t i = 0; i < N; ++i)
        result[i] = out[i] * inv;
    return result;
}

// =====================================================================
// Extirpolate  (real)
// =====================================================================

static void removeIntegerValues(vector<double>& x, vector<double>& y) {
    for (int i = (int)x.size() - 1; i >= 0; --i) {
        if (fmod(x[i], 1.0) == 0.0) {
            x.erase(x.begin() + i);
            y.erase(y.begin() + i);
        }
    }
}

static vector<double> extirpolate(vector<double> x, vector<double> y,
                                   int N, int M) {
    auto p = broadcast_and_flatten(x, y);
    x = p.first;
    y = p.second;

    if (N == 0)
        N = (int)round(*max_element(x.begin(), x.end()) + 0.5 * M + 1);

    vector<double> result(N, 0.0);

    for (size_t i = 0; i < x.size(); ++i) {
        if (fmod(x[i], 1) == 0.0) {
            result[(int)x[i]] += y[i];
        }
    }
    removeIntegerValues(x, y);

    vector<double> ilo = vclip(vfloor(vadd(x, -floor(M / 2.0))), 0, N - M);
    ilo = vmult(ilo, -1);

    vector<double> M_arange = arange(M);
    vector<vector<double>> num_mat(M, vector<double>(x.size()));
    for (int i = 0; i < M; ++i)
        num_mat[i] = vadd(vvadd(x, ilo), -M_arange[i]);

    vector<double> numerator(y.size(), 1.0);
    for (size_t i = 0; i < numerator.size(); ++i) {
        for (int j = 0; j < M; ++j) numerator[i] *= num_mat[j][i];
        numerator[i] *= y[i];
    }

    double denominator = tgamma(M);
    ilo = vmult(ilo, -1);

    for (int j = 0; j < M; j++) {
        if (j > 0) denominator *= (double)j / (j - M);
        vector<double> ind = vadd(ilo, (M - 1 - j));
        for (size_t i = 0; i < ind.size(); ++i) {
            int index = (int)round(ind[i]);
            result[index] += numerator[i] / (denominator * (x[i] - index));
            result[index] = pyint(result[index]);
        }
    }
    return result;
}

// =====================================================================
// Extirpolate  (complex)
// =====================================================================

static void removeIntegerValues_c(vector<double>& x,
                                   vector<complex<double>>& y) {
    for (int i = (int)x.size() - 1; i >= 0; --i) {
        if (fmod(x[i], 1.0) == 0.0) {
            x.erase(x.begin() + i);
            y.erase(y.begin() + i);
        }
    }
}

static vector<complex<double>> extirpolate_complex(
        vector<double> x, vector<complex<double>> y, int N, int M) {
    if (N == 0)
        N = (int)round(*max_element(x.begin(), x.end()) + 0.5 * M + 1);

    vector<complex<double>> result(N, 0.0);

    for (size_t i = 0; i < x.size(); ++i) {
        if (fmod(x[i], 1) == 0.0)
            result[(int)x[i]] += y[i];
    }
    removeIntegerValues_c(x, y);

    vector<double> ilo = vclip(vfloor(vadd(x, -floor(M / 2.0))), 0, N - M);
    ilo = vmult(ilo, -1);

    vector<double> M_arange = arange(M);
    vector<vector<double>> num_mat(M, vector<double>(x.size()));
    for (int i = 0; i < M; ++i)
        num_mat[i] = vadd(vvadd(x, ilo), -M_arange[i]);

    vector<complex<double>> numerator(y.size(), 1.0);
    for (size_t i = 0; i < numerator.size(); ++i) {
        for (int j = 0; j < M; ++j) numerator[i] *= num_mat[j][i];
        numerator[i] *= y[i];
    }

    double denominator = tgamma(M);
    ilo = vmult(ilo, -1);

    for (int j = 0; j < M; j++) {
        if (j > 0) denominator *= (double)j / (j - M);
        vector<double> ind = vadd(ilo, (M - 1 - j));
        for (size_t i = 0; i < ind.size(); ++i) {
            int index = (int)round(ind[i]);
            result[index] += numerator[i] / (denominator * (x[i] - index));
        }
    }
    return result;
}

// =====================================================================
// trig_sum  —  fast trigonometric sum via NFFT
// =====================================================================

static pair<vector<double>, vector<double>>
trig_sum(vector<double> t, vector<double> h,
         double df, int N, double f0, double freq_factor,
         int oversampling = 5, int Mfft = 4) {
    df *= freq_factor;
    f0 *= freq_factor;

    if (df <= 0) throw ValueError("df must be positive");
    if (Mfft <= 0) throw ValueError("Mfft must be positive");

    auto p = broadcast_and_flatten(t, h);
    t = p.first;
    h = p.second;

    unsigned Nfft_temp = (unsigned)bitceil((unsigned long long)N * oversampling);
    int Nfft;
    memcpy(&Nfft, &Nfft_temp, sizeof(int));

    double t0 = *min_element(t.begin(), t.end());
    const complex<double> j2pi(0.0, 2.0 * M_PI);

    if (f0 > 0) {
        vector<complex<double>> exp_exp = vcmult(vadd(t, -t0), j2pi * f0);
        vector<complex<double>> h_complex(h.size());
        for (size_t i = 0; i < h.size(); ++i)
            h_complex[i] = h[i] * exp(exp_exp[i]);

        vector<double> tnorm = vmod(vmult(vadd(t, -t0), Nfft * df), Nfft);
        auto grid = extirpolate_complex(tnorm, h_complex, Nfft, Mfft);
        auto fftgrid = compute_ifft_complex(grid, N);

        if (t0 != 0) {
            vector<double> f = vadd(vmult(arange(N), df), f0);
            auto ee = vcmult(f, j2pi * t0);
            for (size_t i = 0; i < fftgrid.size(); ++i)
                fftgrid[i] *= exp(ee[i]);
        }

        vector<double> S(fftgrid.size()), C(fftgrid.size());
        for (size_t i = 0; i < fftgrid.size(); ++i) {
            C[i] = fftgrid[i].real() * Nfft;
            S[i] = fftgrid[i].imag() * Nfft;
        }
        return {S, C};
    } else {
        vector<double> tnorm = vmod(vmult(vadd(t, -t0), Nfft * df), Nfft);
        auto grid = extirpolate(tnorm, h, Nfft, Mfft);
        auto fftgrid = compute_ifft(grid, N);

        if (t0 != 0) {
            vector<double> f = vadd(vmult(arange(N), df), f0);
            auto ee = vcmult(f, j2pi * t0);
            for (size_t i = 0; i < fftgrid.size(); ++i)
                fftgrid[i] *= exp(ee[i]);
        }

        vector<double> S(fftgrid.size()), C(fftgrid.size());
        for (size_t i = 0; i < fftgrid.size(); ++i) {
            C[i] = fftgrid[i].real() * Nfft;
            S[i] = fftgrid[i].imag() * Nfft;
        }
        return {S, C};
    }
}

// =====================================================================
// Basis term descriptor for the matrix build
// =====================================================================

struct BasisTerm {
    bool is_sin;
    int index;
};

// =====================================================================
// GLS periodogram  —  public C interface
// =====================================================================

extern "C" {

void gls_fast_extern(double* t_pointer,
                     size_t t_size,
                     double* y_pointer,
                     size_t y_size,
                     double* dy_pointer,
                     size_t dy_size,
                     double f0,
                     double df,
                     int Nf,
                     int normalization,
                     bool fit_mean,
                     bool center_data,
                     int nterms,
                     double* output) {

    vector<double> t(t_pointer, t_pointer + t_size);
    vector<double> y(y_pointer, y_pointer + y_size);
    vector<double> dy(dy_pointer, dy_pointer + dy_size);

    vector<double> w = power(dy, -2);
    double ws = vsum(w);

    if (center_data || fit_mean) {
        double dot_prdct = vdot(w, y);
        for (size_t i = 0; i < y.size(); ++i)
            y[i] -= dot_prdct / ws;
    }

    vector<double> yw = vvdivide(y, dy);
    double chi2_ref = vdot(yw, yw);

    // Pre-compute weighted trig sums
    double yws = vsum(vvmult(y, w));

    vector<vector<double>> Sw(2 * nterms + 1, vector<double>(Nf, 0));
    vector<vector<double>> Cw(2 * nterms + 1, vector<double>(Nf, 0));

    for (int i = 0; i < Nf; ++i) Cw[0][i] = ws;

    for (int i = 1; i < 2 * nterms + 1; ++i) {
        auto ts = trig_sum(t, w, df, Nf, f0, (double)i);
        for (int j = 0; j < Nf; ++j) {
            Sw[i][j] = ts.first[j];
            Cw[i][j] = ts.second[j];
        }
    }

    vector<vector<double>> Syw(nterms + 1, vector<double>(Nf, 0));
    vector<vector<double>> Cyw(nterms + 1, vector<double>(Nf, 0));

    for (int i = 0; i < Nf; ++i) Cyw[0][i] = yws;

    vector<double> yw_prod = vvmult(y, w);
    for (int i = 1; i < nterms + 1; ++i) {
        auto ts = trig_sum(t, yw_prod, df, Nf, f0, (double)i);
        for (int j = 0; j < Nf; ++j) {
            Syw[i][j] = ts.first[j];
            Cyw[i][j] = ts.second[j];
        }
    }

    // Build indexing scheme
    vector<BasisTerm> order;
    order.reserve(2 * nterms + (fit_mean ? 1 : 0));

    if (fit_mean) order.push_back({false, 0});
    for (int i = 1; i <= nterms; ++i) {
        order.push_back({true, i});
        order.push_back({false, i});
    }

    size_t order_size = order.size();

    auto getXTX = [&](const BasisTerm& A, const BasisTerm& B,
                       int i) -> double {
        int m = A.index, n = B.index;
        if (A.is_sin && B.is_sin)
            return 0.5 * (Cw[abs(m - n)][i] - Cw[m + n][i]);
        if (!A.is_sin && !B.is_sin)
            return 0.5 * (Cw[abs(m - n)][i] + Cw[m + n][i]);
        if (A.is_sin)
            return 0.5 * (sign(m - n) * Sw[abs(m - n)][i] + Sw[m + n][i]);
        return 0.5 * (sign(n - m) * Sw[abs(n - m)][i] + Sw[n + m][i]);
    };

    auto getXTy = [&](const BasisTerm& A, int i) -> double {
        return A.is_sin ? Syw[A.index][i] : Cyw[A.index][i];
    };

    // Compute power at each frequency
    #pragma omp parallel for
    for (int i = 0; i < Nf; ++i) {
        Eigen::MatrixXd XTX(order_size, order_size);
        Eigen::VectorXd XTy(order_size);

        for (size_t b = 0; b < order_size; ++b) {
            for (size_t a = 0; a < order_size; ++a)
                XTX(b, a) = getXTX(order[a], order[b], i);
            XTy(b) = getXTy(order[b], i);
        }

        output[i] = XTy.dot(XTX.ldlt().solve(XTy));
    }

    // Normalization
    if (normalization == 0) {
        for (int i = 0; i < Nf; ++i) output[i] *= 0.5;
    } else if (normalization == 1) {
        for (int i = 0; i < Nf; ++i) output[i] /= chi2_ref;
    } else if (normalization == 2) {
        for (int i = 0; i < Nf; ++i) output[i] = -log(1.0 - output[i] / chi2_ref);
    } else if (normalization == 3) {
        for (int i = 0; i < Nf; ++i) output[i] = output[i] / (chi2_ref - output[i]);
    }
}

} // extern "C"