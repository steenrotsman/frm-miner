//
// Created by stijn on 6/8/23.
//
#include <vector>
#include <numeric>
#include <cmath>
#include <algorithm>

#include "typing.h"
#include "sax.h"
#include <iostream>
DiscreteDB sax(const TimeSeriesDB& ts, const int seglen, const int alphabet)
{
    DiscreteDB discrete(ts.size());
    for (int i { 0 }; i < ts.size(); i++) {
        discrete[i] = get_row(ts[i], seglen, alphabet);
    }
    return discrete;
}

std::vector<char> get_row(std::vector<double> ts_row, int seglen, int alphabet)
{
    auto rowlen { ceil(static_cast<double>(ts_row.size()) / static_cast<double>(seglen)) };
    std::vector<char> row(static_cast<int>(rowlen));

    for (int j { 0 }; j < rowlen; j++) {
        auto start = ts_row.begin() + j * seglen;
        auto end = start + std::min(seglen, static_cast<int>(ts_row.end() - start));
        double segmean = std::accumulate(start, end, 0.0) /  seglen;
        row[j] = get_discrete_value(alphabet, segmean);
    }

    return row;
}

const std::vector<std::vector<double>> breakpoints {
        {}, {},
        {0},
        {-0.43, 0.43},
        {-0.67, 0, 0.67},
        {-0.84, -0.25, 0.25, 0.84},
        {-0.97, -0.43, 0, 0.43, 0.97},
        {-1.07, -0.57, -0.18, 0.18, 0.57, 1.07},
        {-1.15, -0.67, -0.32, 0, 0.32, 0.67, 1.15},
        {-1.22, -0.76, -0.43, -0.14, 0.14, 0.43, 0.76, 1.22},
        {-1.28, -0.84, -0.52, -0.25, 0, 0.25, 0.52, 0.84, 1.28}
};

char get_discrete_value(int alphabet, double segmean)
{
    char x { 'a' };
    for (auto& breakpoint : breakpoints[alphabet]) {
        if (segmean > breakpoint) {
            x++;
        }
    }
    return x;
}

void znorm(std::vector<std::vector<double>>& ts)
{
    for (auto& series : ts) {
        // Calculate mean
        auto sum { std::accumulate(series.begin(), series.end(), 0.0) };
        auto mean { sum / static_cast<double>(series.size()) };

        // Subtract mean
        std::transform(series.begin(), series.end(), series.begin(), [mean](double x) {
            return x - mean;
        });

        // Calculate stdev - because the mean is 0, the average of series is the average deviation
        double stdev { };
        for (auto val : series) {
            stdev += val * val;
        }
        stdev = sqrt(stdev / static_cast<double>(series.size()));

        // Divide by stdev
        std::transform(series.begin(), series.end(), series.begin(), [stdev](double x) {
            return x / stdev;
        });
    }
}