//
// Created by fabian on 9/4/24.
//

#include "maths.h"
#include "cmath"
#include "vector"
#include "vector_operations.h"
#include <random>
#include <stdexcept>
#include <numeric>

using namespace std;

double sinusoid(double x, double amplitude, double period, double offset, double phase) {
    // The general form of the sinusoid is:
    // y(x) = amplitude * sin(2 * M_PI * (x / period + phase)) + offset

    double result = amplitude * sin(2 * M_PI * (x / period + phase)) + offset;
    return result;
}

vector<double> sinusoid_wrapper(vector<double> x, double amplitude, double period, double offset, double phase, double Noise){
    vector<double> y(x.size());
    if (Noise == 0){
        for (int i = 0; i < x.size(); ++i) {
            y[i] = sinusoid(x[i], amplitude, period, offset, phase);
        }
    }
    else{
        random_device rd;
        mt19937 gen(rd());
        normal_distribution<> dist(0.0, Noise);
        for (int i = 0; i < x.size(); ++i) {
            y[i] = sinusoid(x[i], amplitude, period, offset, phase) + dist(gen);
        }
    }

    return y;
}

vector<double> genOptimalPeriodogramSamples(const vector<double>& t, double sample_factor, double forced_min_p, double forced_max_p, int forced_N){
    vector<double> result(3);
    double min_p;
    double max_p;
    if (forced_min_p == 0){
        min_p = minimum_diff(t)*2;
    }
    else {
        min_p = forced_min_p;
    }
    if (forced_max_p == 0){
        max_p = (get_max(t)-get_min(t))/2;
    }
    else {
        max_p = forced_max_p;
    }
    double x_ptp = get_max(t)-get_min(t);
    double Npoints;
    double df;
    if (not forced_N){
        double n = ceil(x_ptp / min_p);
        double R_p = (x_ptp / (n - 1) - x_ptp / n);
        df = (1 / min_p - (1 / (min_p + R_p)))/sample_factor;

        Npoints = ceil((1 / min_p - 1 / max_p) / df);
    }
    else{
        df = (1/min_p-1/max_p)/forced_N;
        Npoints = (double)forced_N;
    }

    result[0] = 1/max_p;
    result[1] = df;
    result[2] = Npoints;
    return result;
}



vector<double> calculateLogBinEdges(const vector<double>& binCenters) {
    if (binCenters.size() < 2) {
        throw invalid_argument("There must be at least two bin centers.");
    }

    vector<double> binEdges(binCenters.size() + 1);

    // Calculate the first bin edge (lower edge of the first bin)
    double R_first = binCenters[1] / binCenters[0];
    binEdges[0] = binCenters[0] / sqrt(R_first);

    // Calculate the internal bin edges
    for (size_t i = 1; i < binCenters.size(); ++i) {
        double R = binCenters[i] / binCenters[i - 1];
        binEdges[i] = binCenters[i - 1] * sqrt(R);
    }

    // Calculate the upper edge of the last bin
    double R_last = binCenters.back() / binCenters[binCenters.size() - 2];
    binEdges.back() = binCenters.back() * sqrt(R_last);

    return binEdges;
}


vector<double> binCenterFromEdges(vector<double> binEdges, bool islog){
    if (islog){
        for (double & binEdge : binEdges) {
            binEdge = log10(binEdge);
        }
    }
    vector<double> binCenters(binEdges.size()-1);
    for (int i = 0; i < binEdges.size()-1; ++i) {
        binCenters[i] = (binEdges[i+1]-binEdges[i])/2;
    }
    if (islog){
        for (double binCenter: binCenters) {
            binCenter = pow(10, binCenter);
        }
    }
    return binCenters;
}


vector<double> randomValuesFromBins(const vector<double>& binCenters, const vector<double>& weights, size_t N, bool use_seed, unsigned int seed, bool log_spaced, bool add_randomness) {
    if (binCenters.size() != weights.size() || binCenters.size() < 2) {
        throw invalid_argument("Bin centers and weights must be of the same size and at least 2 bins.");
    }

    // Compute the bin edges
    vector<double> binEdges(binCenters.size() + 1);
    if (!log_spaced){
        binEdges[0] = binCenters[0] - 0.5 * (binCenters[1] - binCenters[0]);
        for (size_t i = 1; i < binCenters.size(); ++i) {
            binEdges[i] = 0.5 * (binCenters[i - 1] + binCenters[i]);
        }
        binEdges.back() = binCenters.back() + 0.5 * (binCenters.back() - binCenters[binCenters.size() - 2]);
    }
    else {
        binEdges = calculateLogBinEdges(binCenters);
    }

    // Initialize random number generator
    mt19937 gen;
    if (use_seed) {
        gen.seed(seed);
    } else {
        random_device rd;
        gen.seed(rd());
    }

    // Create a distribution to pick a bin based on weights
    discrete_distribution<> binDist(weights.begin(), weights.end());

    // Vector to store generated values
    vector<double> randomValues;
    randomValues.reserve(N);  // Reserve space for N values

    // Generate N random values
    for (size_t i = 0; i < N; ++i) {
        // Pick a bin
        int chosenBin = binDist(gen);
        if (add_randomness){
        // Generate a uniform random number within the chosen bin
            if (!log_spaced){
                uniform_real_distribution<> valueDist(binEdges[chosenBin], binEdges[chosenBin + 1]);
                randomValues.push_back(valueDist(gen));
            }
            else {
                uniform_real_distribution<> valueDist(log10(binEdges[chosenBin]), log10(binEdges[chosenBin + 1]));
                randomValues.push_back(pow(10, (valueDist(gen))));
            }
        }
        else {
            randomValues.push_back((binEdges[chosenBin]+binEdges[chosenBin + 1])/2);
        }
    }
    return randomValues;
}


double randomValueInRange(double lo, double hi, bool use_seed, unsigned seed){
    // Create a random number generator
    mt19937 rng; // Mersenne Twister random generator

    // Use seed if provided
    if(use_seed) {
        rng.seed(seed);
    } else {
        random_device rd; // Non-deterministic random device
        rng.seed(rd());
    }

    // Create a uniform distribution in the range [lo, hi]
    uniform_real_distribution<double> dist(lo, hi);

    // Return a random value in the range
    return dist(rng);
}


tuple<vector<double>, vector<double>, vector<double>, vector<double>> sinusoidMonteCarlo(const vector<double>& x, const vector<double>& y, const vector<double>& y_err, double period, int N_sim, double amp_0, double phase_0, double offset_0, double amp_step, double phase_step, double offset_step, int N_burn_in){
    // Random number generation
    random_device rd;
    mt19937 gen(rd());
    normal_distribution<> dist(0.0, 1.0);
    uniform_real_distribution<> dist_uni(0.0, 1.0);

    vector<double> amp_result   (N_sim-N_burn_in);
    vector<double> phase_result (N_sim-N_burn_in);
    vector<double> offset_result(N_sim-N_burn_in);
    vector<double> chisq_result (N_sim-N_burn_in);


    auto chiSquared = [&](double amp, double phase, double offset) {
        double chi2 = 0.0;
        for (size_t i = 0; i < x.size(); ++i) {
            double model = sinusoid(x[i], amp, period, offset, phase);
            chi2 += pow((y[i] - model) / y_err[i], 2);
        }
        return chi2;
    };

    double chi2_current = chiSquared(amp_0, phase_0, offset_0);

    // Monte Carlo loop
    for (int i = 0; i < N_sim; ++i) {
        // Propose new parameters by perturbing the current guess
        double amp_proposed = amp_0 + amp_step * dist(gen);
        double phase_proposed = fmod(phase_0 + phase_step * dist(gen), 1);
        double offset_proposed = offset_0 + offset_step * dist(gen);

        // Calculate new chi-squared with proposed parameters
        double chi2_proposed = chiSquared(amp_proposed, phase_proposed, offset_proposed);

        // Metropolis-Hastings acceptance criterion
        if (chi2_proposed < chi2_current || exp(-(chi2_proposed - chi2_current) / 2) > dist_uni(gen)) {
            // Accept the new parameters
            amp_0 = amp_proposed;
            phase_0 = phase_proposed;
            offset_0 = offset_proposed;
            chi2_current = chi2_proposed;
            if (i >= N_burn_in){
                amp_result[i-N_burn_in] = amp_0;
                phase_result[i-N_burn_in] = phase_0;
                offset_result[i-N_burn_in] = offset_0;
                chisq_result[i-N_burn_in] = chi2_current;
            }
        }
        else if (i >= N_burn_in){
            amp_result[i-N_burn_in] = amp_0;
            phase_result[i-N_burn_in] = phase_0;
            offset_result[i-N_burn_in] = offset_0;
            chisq_result[i-N_burn_in] = chi2_current;
        }
    }
    return make_tuple(amp_result, phase_result, offset_result, chisq_result);
}


// Constants
const double EPSILON = 1e-10;
const double MAX_ITER = 1000;

// Gamma function approximation (using Lanczos approximation or other)
double gamma_function(double z) {
    // Implementation of the Gamma function
    if (z <= 0) {
        throw std::invalid_argument("Invalid input to Gamma function");
    }
    // Use a library like Boost for a more accurate implementation
    return tgamma(z);
}

// Incomplete gamma function (lower incomplete gamma)
double gamma_incomplete(double a, double x) {
    double sum = 1.0 / a;
    double value = sum;
    for (int n = 1; n < MAX_ITER; n++) {
        sum *= x / (a + n);
        value += sum;
        if (sum < EPSILON * value) break;
    }
    return value * std::exp(-x + a * std::log(x) - std::log(gamma_function(a)));
}

// Chi-squared Q-function implementation
double gsl_cdf_chisq_Q(double x, double nu) {
    if (x < 0) {
        throw std::invalid_argument("x must be non-negative");
    }
    if (nu <= 0) {
        throw std::invalid_argument("Degrees of freedom must be positive");
    }

    // Q(x; nu) = 1 - P(x; nu)
    // P(x; nu) is the CDF of chi-squared, equivalent to the regularized gamma function
    double P = gamma_incomplete(nu / 2.0, x / 2.0);
    return 1.0 - P;
}



double vrad_pvalue(const vector<double>& vrad, const vector<double>& vrad_err) {
    int ndata = vrad.size();
    if (ndata < 2) {
        return NAN;
    }
    int nfit = 1;

    // Calculate weighted mean of vrad
    double weighted_sum = 0.0, weight_sum = 0.0;
    for (int i = 0; i < ndata; ++i) {
        weighted_sum += vrad[i] / vrad_err[i];
        weight_sum += 1.0 / vrad_err[i];
    }
    double vrad_wmean = weighted_sum / weight_sum;

    // Calculate chi values
    vector<double> chi(ndata);
    for (int i = 0; i < ndata; ++i) {
        chi[i] = (vrad[i] - vrad_wmean) / vrad_err[i];
    }

    // Calculate chisq_sum
    double chisq_sum = 0.0;
    for (double val : chi) {
        chisq_sum += val * val;
    }

    // Calculate degrees of freedom
    double dof = ndata - nfit;

    // Calculate p-value using chi-squared survival function (sf)
    double pval = gsl_cdf_chisq_Q(chisq_sum / 2, (dof / 2));
    double logp = log10(pval);

    if (pval == 0) {
        return -500;
    }
    if (isnan(logp)) {
        return 0;
    }

    return logp;
}
