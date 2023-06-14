//
// Created by stijn on 6/8/23.
//
#include "sax.h"
#include <vector>
#include <numeric>
#include <algorithm>
#include <cmath>

std::vector<std::vector<int>> sax(std::vector<std::vector<double>>& ts, int seglen, int alphabet)
{
    std::vector<std::vector<int>> discrete(ts.size());
    for (size_t i { 0 }; i < ts.size(); i++) {
        auto rowlen { ceil(static_cast<double>(ts[i].size()) / static_cast<double>(seglen)) };
        std::vector<int> row(static_cast<int>(rowlen));

        for (size_t j { 0 }; j < rowlen; j++) {
            double segsum { std::accumulate(&ts[i][j*seglen], &ts[i][j*seglen+1], 0.0) };
            double segmean { segsum / seglen };
            row[j] = get_discrete_value(alphabet, segmean);
        }
        discrete[i] = row;
    }
    return discrete;
}

std::vector<std::vector<double>> breakpoints {
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

int get_discrete_value(int alphabet, double segmean)
{
    int x { 0 };
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

        // Calculate stdev
        double sq_sum { std::inner_product(series.begin(), series.end(), series.begin(), 0.0) };
        double stdev { std::sqrt(sq_sum / static_cast<double>(series.size())) };

        // Divide by stdev
        std::transform(series.begin(), series.end(), series.begin(), [stdev](double x) {
            return x / stdev;
        });
    }
}