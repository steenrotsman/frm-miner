//
// Created by stijn on 6/8/23.
//

#include <utility>
#include <vector>
#include <map>
#include <cmath>

#include "motif.h"
#include "typing.h"

Motif::Motif(Pattern pattern) : pattern(std::move(pattern)), length(0), naed(0.0) {}

void Motif::record_index(int seq, int idx_in_seq)
{
    indexes[seq].push_back(idx_in_seq);
}

void Motif::map(const TimeSeriesDB& timeseries, const int seglen)
{
    length = static_cast<int>(pattern.size()) * seglen;
    representative.resize(length);

    set_average_occurrences(timeseries);
    set_representative();
    set_best_matches_and_rmse(timeseries);
}

void Motif::set_average_occurrences(const TimeSeriesDB &timeseries)
{
    for (auto& [ ts, idx ] : indexes) {
        average_occurrences[ts] = std::vector<double>(length);
        for (auto id : idx) {
            for (int i { 0 }; i < length; i++) {
                average_occurrences[ts][i] += timeseries[ts][id+i] / static_cast<double>(idx.size());
            }
        }
    }
}

void Motif::set_representative()
{

    for (auto& [ts, average_occurrence] : average_occurrences) {
        for (int i { 0 }; i < length; i++) {
            representative[i] += average_occurrence[i] / static_cast<double>(average_occurrences.size());
        }
    }
}

void Motif::set_best_matches_and_rmse(const TimeSeriesDB &timeseries)
{
    for (auto& [ts, idx] : indexes) {
        double min_dist {100.0};
        int best_match {};
        double distance {};

        for (auto id : idx) {
            distance = 0.0;

            // Calculate ED of occurrence
            for (int i { 0 }; i < length; i++) {
                distance += pow(timeseries[ts][id+i] - representative[i], 2);
            }
            distance = sqrt(distance);

            // Update bsf distance
            if (distance < min_dist) {
                min_dist = distance;
                best_match = id;
            }
        }

        naed += min_dist;
        best_matches[ts] = best_match;
    }
    naed /= static_cast<double>(indexes.size()) * static_cast<double>(length);
}