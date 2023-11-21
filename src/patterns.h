//
// Created by stijn on 6/8/23.
//

#ifndef FRM_C_PATTERNS_H
#define FRM_C_PATTERNS_H

#include <map>
#include <vector>

#include "typing.h"
#include "motif.h"

class PatternMiner {
private:
    double minsup;
    int min_len;
    int max_len;
    double max_overlap;
    std::vector<Pattern> frequent;
    double min_freq;
    int k;
    std::map<Pattern, Motif> k_motifs;
    std::map<Pattern, Motif> k_1_motifs;

    void mine_1_patterns(const DiscreteDB& sequences);
    std::vector<Pattern> get_candidates();
    void find_occurrences(const Pattern& candidate, const DiscreteDB& sequences, Motif& motif);
    void remove_redundant();
public:
    PatternMiner(double minsup, int min_len, int max_len, double max_overlap);
    void mine(const DiscreteDB& sequences);
    std::vector<Pattern> get_frequent() { return frequent; };
};

bool is_p_in_vec(const Pattern& p, const std::vector<Pattern>& vec);
double lcs(const Pattern& p1, const Pattern& p2);

#endif //FRM_C_PATTERNS_H
