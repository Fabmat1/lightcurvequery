//
// Created by fabian on 9/4/24.
//

#ifndef SUBDWARF_RV_SIMULATION_FILE_IO_CPP_H
#define SUBDWARF_RV_SIMULATION_FILE_IO_CPP_H

#endif //SUBDWARF_RV_SIMULATION_FILE_IO_CPP_H
#include <vector>
#include <string>
using namespace std;

vector<vector<double>> readCSV(const string& filename, bool skipheader=false, const string& delimiter=",");
void saveCSV(const string& filename, const vector<vector<double>>& data);