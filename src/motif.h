//
// Created by stijn on 6/8/23.
//

#ifndef FRM_C_MOTIF_H
#define FRM_C_MOTIF_H

#include <vector>
#include <map>
#include <unordered_map>

#include "typing.h"

class Motif {
private:
    Pattern pattern;
    int _seglen;
    int length;
    std::unordered_map<int, std::vector<int>> indexes;
    std::unordered_map<int, std::vector<double>> average_occurrences;
    std::vector<double> representative;
    std::unordered_map<int, int> best_matches;
    double naed;

    std::vector<double> get_occurrence(const std::vector<double>& ts, int index) const;
    void set_average_occurrences(const TimeSeriesDB& timeseries);
    void set_representative();
    void set_best_matches_and_naed(const TimeSeriesDB &timeseries);
public:
    explicit Motif(Pattern pattern);

    void record_index(int seq, int idx_in_seq);

    Pattern get_pattern() { return pattern; };
    std::unordered_map<int, std::vector<int>> get_indexes() { return indexes; };
    std::unordered_map<int, std::vector<double>> get_average_occurrences() { return average_occurrences; };
    std::vector<double> get_representative() { return representative; };
    std::unordered_map<int, int> get_best_matches() { return best_matches; };
    int get_length() const { return length; };
    double get_naed() const { return naed; };

    void map(const TimeSeriesDB& timeseries, int seglen);
};

#endif //FRM_C_MOTIF_H
