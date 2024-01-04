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
        auto len = std::min(seglen, static_cast<int>(ts_row.end() - start));
        auto end = start + len;
        double segmean = std::accumulate(start, end, 0.0) /  len;
        row[j] = get_discrete_value(alphabet, segmean);
    }

    return row;
}

const std::vector<std::vector<double>> breakpoints {
        {}, {},
        {0},
        {-0.4307273, 0.4307273},
        {-0.67448975, 0, 0.67448975},
        {-0.84162123, -0.2533471,  0.2533471,  0.84162123},
        {-0.96742157, -0.4307273,  0, 0.4307273, 0.96742157},
        {-1.0675705238781414, -0.5659488219328631, -0.1800123697927051, 0.18001236979270496, 0.5659488219328631, 1.0675705238781412},
        {-1.1503493803760079, -0.6744897501960817, -0.31863936396437514, 0.0, 0.31863936396437514, 0.6744897501960817, 1.1503493803760079},
        {-1.22064034884735, -0.7647096737863871, -0.43072729929545756, -0.13971029888186212, 0.13971029888186212, 0.43072729929545744, 0.7647096737863871, 1.2206403488473496},
        {-1.2815515655446004, -0.8416212335729142, -0.5244005127080409, -0.2533471031357997, 0.0, 0.2533471031357997, 0.5244005127080407, 0.8416212335729143, 1.2815515655446004},

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