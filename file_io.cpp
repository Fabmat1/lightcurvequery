//
// Created by fabian on 9/4/24.
//

#include <iostream>
#include <fstream>
#include <sstream>
#include <string>
#include <tuple>
#include <vector>
#include <iomanip>


using namespace std;

vector<vector<double>> readCSV(const string& filename, bool skipheader, const string& delimiter) {
    vector<vector<double>> columns;
    ifstream file(filename);

    if (!file.is_open()) {
        cerr << "readCSV: Failed to open the file: "<< filename << endl;
        return columns; // return empty vector in case of error
    }

    string line;
    int n = 0;
    while (getline(file, line)) {
        if(skipheader && n == 0){
            n++;
            continue;
        }
        n++;
        stringstream ss(line);
        string value;
        vector<double> rowValues;

        // Read values from the line and convert them to double
        while (getline(ss, value, *delimiter.c_str())) {
            rowValues.push_back(stod(value));
        }

        // Resize columns if this is the first row
        if (columns.empty()) {
            columns.resize(rowValues.size());
        }

        // Add values to respective column vectors
        for (size_t i = 0; i < rowValues.size(); ++i) {
            columns[i].push_back(rowValues[i]);
        }
    }

    file.close();
    return columns;
}



void saveCSV(const string& filename, const vector<vector<double>>& data) {
    ofstream file(filename);

    if (!file.is_open()) {
        cerr << "saveCSV: Failed to open the file." << endl;
        return;
    }

    // Determine the number of rows and columns
    size_t numRows = data.empty() ? 0 : data[0].size();
    size_t numCols = data.size();

    for (size_t i = 0; i < numRows; ++i) {
        for (size_t j = 0; j < numCols; ++j) {
            file << setprecision(15) << data[j][i]; // Write the value

            // Add a comma if this is not the last column
            if (j < numCols - 1) {
                file << ",";
            }
        }
        file << endl; // New line after each row
    }

    file.close();
    cout << "Data saved to " << filename << endl;
}