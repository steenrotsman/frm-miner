//
// Created by stijn on 6/8/23.
//

#ifndef FRM_C_SAX_H
#define FRM_C_SAX_H

#include <vector>

#include "typing.h"

DiscreteDB sax(const TimeSeriesDB & ts, int seglen, int alphabet);
std::vector<char> get_row(std::vector<double> ts_row, int seglen, int alphabet);
char get_discrete_value(int alphabet, double segmean);
void znorm(TimeSeriesDB& ts);


#endif //FRM_C_SAX_H
