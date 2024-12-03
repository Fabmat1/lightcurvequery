//
// Created by fabian on 9/1/24.
//

#ifndef SUBDWARF_RV_SIMULATION_EXCEPTIONS_H
#define SUBDWARF_RV_SIMULATION_EXCEPTIONS_H

#endif //SUBDWARF_RV_SIMULATION_EXCEPTIONS_H
#include <string>
#include <exception>
using namespace std;

class ValueError : public exception
{
public:
    explicit ValueError(string  msg);

    ~ValueError() override;

    [[nodiscard]] const char* what() const noexcept override;

private:
    const string m_msg;
};