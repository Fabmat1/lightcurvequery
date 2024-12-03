//
// Created by fabia on 08.09.2024.
//

#include <numeric>
#include "models.h"
#include "maths.h"
#include "vector_operations.h"
#include "algorithm"
#include "iostream"
#include "vector"
#include "cmath"
#include "omp.h"
#include "file_io.h"
#include "lomb_scargle_periodogram.h"
#include "chrono"


using namespace std;

void normalize_histogram(vector<vector<double>>& histogram) {
    // Calculate the total sum of the histogram
    double total_sum = 0.0;
    for (const auto& row : histogram) {
        total_sum += accumulate(row.begin(), row.end(), 0.0);
    }

    // Normalize the histogram by dividing each element by the total sum
    if (total_sum > 0) {
        for (auto &row: histogram) {
            for (auto &value: row) {
                value /= total_sum;
            }
        }
    }
}


void Star::calculate_orbit_prediciton(int Nx, int Ny, double amp_lim, double offset_lim){
    if (this->periodogram_x.empty()  || this->periodogram_y.empty()){
        return;
    }

    this->period_amp_histogram = vector<vector<double>>(Nx, vector<double>(Ny, 0.));
    this->period_offset_histogram = vector<vector<double>>(Nx, vector<double>(Ny, 0.));
    this->period_phase_histogram = vector<vector<double>>(Nx, vector<double>(Ny, 0.));

    double min_p = *min_element(begin(this->periodogram_x), end(this->periodogram_x));
    double max_p = *max_element(begin(this->periodogram_x), end(this->periodogram_x));

    vector<double> period_edges = linspace(log10(min_p), log10(max_p), Nx+1);
    transform(period_edges.begin(), period_edges.end(), period_edges.begin(), [](double x){
        return pow(10, x);
    });


    vector<double> amp_edges = linspace(0, amp_lim, Ny+1);
    vector<double> phase_edges = linspace(0, 1, Ny+1);
    vector<double> offset_edges = linspace(-offset_lim, offset_lim, Ny+1);
    transform(this->periodogram_y.begin(), this->periodogram_y.end(), this->periodogram_y.begin(),
              [](double x) { return x < 0.0 ? 0.0 : x; });
    this->periodogram_y = vdivide(this->periodogram_y, vsum(this->periodogram_y));

    vector<double> period_chi_sums(Nx);
    int total_progress = 0;
    int print_every = 1000;

    auto start_time = std::chrono::high_resolution_clock::now();

    #pragma omp parallel for schedule(static)
    for (int j = 0; j < this->periodogram_x.size(); ++j) {
        #pragma omp atomic
        total_progress++;
        if (total_progress % print_every == 0) {
            #pragma omp critical
                {
                    auto current_time = std::chrono::high_resolution_clock::now();
                    std::chrono::duration<double, std::milli> elapsed = current_time - start_time;
                    double avg_speed_ms_per_loop = elapsed.count() / total_progress;
                    cout << "Progress: " << total_progress << " out of " << this->periodogram_x.size()
                         << ". Avg. speed: " << avg_speed_ms_per_loop << " ms/loop" << endl;
                }
            }
        //amp_result, phase_result, offset_result, chisq_result
        tuple<vector<double>, vector<double>, vector<double>, vector<double>>
        curr_period_results = sinusoidMonteCarlo(this->samples,
                           this->datapoints,
                           this->datapoint_errors,
                           this->periodogram_x[j],
                           25000,
                           ptp(datapoints)/2,
                           .5,
                           vsum(datapoints)/datapoints.size());

        // Find the period bin index (x-axis)
        auto period_bin = lower_bound(period_edges.begin(), period_edges.end(), this->periodogram_x[j]) - period_edges.begin() - 1;
        const auto& amps = get<0>(curr_period_results);
        const auto& phases = get<1>(curr_period_results);
        const auto& offsets = get<2>(curr_period_results);
        const auto& chisqs = get<3>(curr_period_results);

        for (int i = 0; i < amps.size(); ++i) {
            double curr_amp = amps[i];
            double curr_phase = fmod(phases[i], 1);
            double curr_offset = offsets[i];
            double curr_chisq = chisqs[i];

            // Find the corresponding bins for amplitude, phase, and offset (y-axes)
            auto amp_bin = lower_bound(amp_edges.begin(), amp_edges.end(), curr_amp) - amp_edges.begin() - 1;
            auto phase_bin = lower_bound(phase_edges.begin(), phase_edges.end(), curr_phase) - phase_edges.begin() - 1;
            auto offset_bin = lower_bound(offset_edges.begin(), offset_edges.end(), curr_offset) - offset_edges.begin() - 1;

            // Check if bins are valid and update the histograms
            if (period_bin >= 0 && period_bin < Nx) {
                if (amp_bin >= 0 && amp_bin < Ny) {
                    #pragma omp atomic
                    this->period_amp_histogram[period_bin][amp_bin] += periodogram_y[j]/curr_chisq;
                }
                if (phase_bin >= 0 && phase_bin < Ny) {
                    #pragma omp atomic
                    this->period_phase_histogram[period_bin][phase_bin] += periodogram_y[j]/curr_chisq;
                }
                if (offset_bin >= 0 && offset_bin < Ny) {
                    #pragma omp atomic
                    this->period_offset_histogram[period_bin][offset_bin] += periodogram_y[j]/curr_chisq;
                }
            }
        }
        if (period_bin >= 0 && period_bin < Nx){
            #pragma omp atomic
            period_chi_sums[period_bin] += vsum(chisqs);
        }
    }
    cout << "coming out of the loop..." << endl;

    for (int i = 0; i < period_chi_sums.size(); ++i) {
        period_amp_histogram[i] = vdivide(this->period_amp_histogram[i], period_chi_sums[i]);
        period_offset_histogram[i] = vdivide(this->period_offset_histogram[i], period_chi_sums[i]);
        period_phase_histogram[i] = vdivide(this->period_phase_histogram[i], period_chi_sums[i]);
    }

    cout << "Normalizing..." << endl;

    normalize_histogram(this->period_amp_histogram);
    normalize_histogram(this->period_offset_histogram);
    normalize_histogram(this->period_phase_histogram);
}

void Star::process_star(int index, double forced_min_p, double forced_max_p, int forced_N){
    vector<double> opt_pgram_params = genOptimalPeriodogramSamples(this->samples, 20, forced_min_p, forced_max_p, forced_N);
    if (opt_pgram_params[2] < 100000){
        opt_pgram_params[1] *= opt_pgram_params[2]/100000;
        opt_pgram_params[2] = 100000;
    }
    cout << "MC Simulation from " << 1/(opt_pgram_params[0]+opt_pgram_params[1]*opt_pgram_params[2]) << "d to " << 1/opt_pgram_params[0] << "d while using " << opt_pgram_params[2] << " Samples." << endl;

    this->periodogram_y = gls_fast(this->samples, this->datapoints, this->datapoint_errors,
                                  opt_pgram_params[0], opt_pgram_params[1], (int)round(opt_pgram_params[2]));
    vector<double> pgram_x = linspace(opt_pgram_params[0], opt_pgram_params[0]+opt_pgram_params[2]*opt_pgram_params[1], (int)ceil(opt_pgram_params[2]));
    this->periodogram_x = invert(pgram_x);
    vector<vector<double>> outdata;
    outdata.push_back(this->periodogram_x);
    outdata.push_back(this->periodogram_y);
    saveCSV("../out/"+to_string(index)+".csv", outdata);

    cout << "Period:    "<< this->period << endl;
    cout << "Amplitude: "<< this->amplitude << endl;
    cout << "Offset:    "<< this->offset << endl;
    cout << "Phase:     "<< this->phase << endl;

    this->calculate_orbit_prediciton(5000, 1000, 500., 500.);

    saveCSV("../out/pamp"+to_string(index)+".csv", this->period_amp_histogram);
    saveCSV("../out/pphase"+to_string(index)+".csv", this->period_phase_histogram);
    saveCSV("../out/poffset"+to_string(index)+".csv", this->period_offset_histogram);

    vector<vector<double>> starrv;
    starrv.push_back(this->samples);
    starrv.push_back(this->datapoints);
    starrv.push_back(this->datapoint_errors);

    saveCSV("../out/rvs"+to_string(index)+".csv", starrv);
}

void Star::sort_samples() {
    // Create an index vector with the original indices
    vector<size_t> indices(this->samples.size());
    for (size_t i = 0; i < indices.size(); ++i) {
        indices[i] = i;
    }

    // Sort the indices based on the corresponding values in this->samples
    sort(indices.begin(), indices.end(), [&](size_t a, size_t b) {
        return this->samples[a] < this->samples[b];
    });

    // Use the sorted indices to create sorted copies of the vectors
    vector<double> sorted_samples(this->samples.size());
    vector<double> sorted_datapoints(this->datapoints.size());
    vector<double> sorted_datapoint_errors(this->datapoint_errors.size());

    for (size_t i = 0; i < indices.size(); ++i) {
        sorted_samples[i] = this->samples[indices[i]];
        sorted_datapoints[i] = this->datapoints[indices[i]];
        sorted_datapoint_errors[i] = this->datapoint_errors[indices[i]];
    }

    // Overwrite the original vectors with the sorted ones
    this->samples = std::move(sorted_samples);
    this->datapoints = std::move(sorted_datapoints);
    this->datapoint_errors = std::move(sorted_datapoint_errors);
}