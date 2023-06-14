//
// Created by stijn on 6/8/23.
//

#ifndef FRM_C_SAX_H
#define FRM_C_SAX_H

#include <vector>

#include "typing.h"

DiscreteDB sax(TimeSeriesDB & ts, int seglen, int alphabet);
int get_discrete_value(int alphabet, double segmean);
void znorm(std::vector<std::vector<double>>& ts);


#endif //FRM_C_SAX_H