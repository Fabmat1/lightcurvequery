//
// Created by fabian on 9/1/24.
//

#include "exceptions.h"
#include <iostream>
#include <string>
#include <utility>

using namespace std;



ValueError::ValueError(string  msg) : m_msg(std::move(msg))
{
    cout << "ValueError::ValueError - set m_msg to:" << m_msg << endl;
}

ValueError::~ValueError()
{
    cout << "ValueError::~ValueError" << endl;
}

const char* ValueError::what() const noexcept
{
    cout << "ValueError::what" << endl;
    return m_msg.c_str();
}
