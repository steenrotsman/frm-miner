//
// Created by stijn on 6/8/23.
//

#include <utility>
#include <vector>
#include <map>
#include <cmath>

#include "motif.h"
#include "typing.h"

Motif::Motif(Pattern pattern) : pattern(std::move(pattern)), _seglen(), length(0), naed(0.0) {}

void Motif::record_index(int seq, int idx_in_seq)
{
    indexes[seq].push_back(idx_in_seq);
}

void Motif::map(const TimeSeriesDB& timeseries, const int seglen)
{
    _seglen = seglen;
    length = static_cast<int>(pattern.size()) * _seglen;
    representative.resize(length);

    set_average_occurrences(timeseries);
    set_representative();
    set_best_matches_and_naed(timeseries);
}

std::vector<double> Motif::get_occurrence(const std::vector<double>& ts, int index) const
{
    int start = index * _seglen;
    int end = start + length;
    std::vector<double> occurrence;
    occurrence.reserve(length);

    // Check if the time series is too short to fit the entire occurrence
    int too_short { std::max(0, end - static_cast<int>(ts.size())) };

    // Fill the occurrence with items from the time series
    for (int i = start; i < end - too_short; i++) {
        occurrence.push_back(ts[i]);
    }

    // If the time series is too short, make the rest of the occurrence NaNs
    for (int i = 0; i < too_short; i++) {
        occurrence.push_back(std::nan(""));
    }

    return occurrence;
}

void Motif::set_average_occurrences(const TimeSeriesDB &timeseries)
{
    for (const auto& [ ts, idx ] : indexes) {
        std::vector<std::vector<double>> occurrences;

        for (int id : idx) {
            std::vector<double> occurrence = get_occurrence(timeseries[ts], id);
            occurrences.push_back(occurrence);
        }

        std::vector<double> average_occurrence;
        int numOccurrences = static_cast<int>(occurrences.size());
        int numElements = static_cast<int>(occurrences[0].size());

        for (int i = 0; i < numElements; i++) {
            double sum = 0.0;
            int count = 0;
            for (int j = 0; j < numOccurrences; j++) {
                if (!std::isnan(occurrences[j][i])) {
                    sum += occurrences[j][i];
                    count++;
                }
            }
            if (count > 0) {
                double avg = sum / static_cast<double>(count);
                average_occurrence.push_back(avg);
            } else {
                average_occurrence.push_back(std::nan(""));
            }
        }

        average_occurrences[ts] = average_occurrence;
    }
}

void Motif::set_representative()
{
    for (const auto& [ts, average_occurrence] : average_occurrences) {
        std::vector<double> valid_values;
        int numOccurrences = static_cast<int>(average_occurrences.size());

        for (int i = 0; i < length; i++) {
            double sum = 0.0;
            int count = 0;

            for (const auto& occurrence : average_occurrences) {
                double value = occurrence.second[i];
                if (!std::isnan(value)) {
                    sum += value;
                    count++;
                }
            }

            if (count > 0) {
                double avg = sum / static_cast<double>(count);
                valid_values.push_back(avg);
            } else {
                // If all occurrences have NaN in this column, set representative to NaN.
                valid_values.push_back(std::nan(""));
            }
        }

        for (int i = 0; i < length; i++) {
            representative[i] += valid_values[i] / static_cast<double>(numOccurrences);
        }
    }
}

void Motif::set_best_matches_and_naed(const TimeSeriesDB& timeseries)
{
    for (const auto& [ts, idx] : indexes) {
        int best_match = 0;
        double min_naed = 1.0e6;

        for (int id : idx) {
            // Get the occurrence from the time series
            std::vector<double> occurrence = get_occurrence(timeseries[ts], id);

            // Calculate NAED of occurrence, excluding NaN values
            double dist { };
            int valid_values_count = 0;
            for (int i = 0; i < length; i++) {
                if (!std::isnan(occurrence[i])) {
                    dist += pow(occurrence[i] - representative[i], 2);
                    valid_values_count++;
                }
            }
            dist = sqrt(dist);

            // Update min_naed and best_match
            if (dist < min_naed) {
                min_naed = dist;
                best_match = id;
            }
        }

        naed += min_naed;
        best_matches[ts] = best_match * _seglen;
    }

    naed /= static_cast<double>(indexes.size()) * static_cast<double>(length);
}