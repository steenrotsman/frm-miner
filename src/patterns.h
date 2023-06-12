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
    long min_len;
    double max_overlap;
    std::map<Pattern, Motif> frequent;
    double min_freq;
    int k;
    std::vector<std::vector<Pattern>> patterns;

    void mine_1_patterns(DiscreteDB& sequences);
    void prune_infrequent(const Pattern& pattern);
    std::vector<Pattern> get_candidates();
    void find_candidate(Pattern candidate, DiscreteDB& sequences);
    void remove_redundant();
    std::vector<Pattern> remove_short();
public:
    PatternMiner(double minsup, long min_len, double max_overlap);
    void mine(DiscreteDB& sequences);
    std::map<Pattern, Motif> get_frequent() { return frequent; };
};

bool is_p_in_vec(const Pattern& p, const std::vector<Pattern>& vec);
double lcs(const Pattern& p1, const Pattern& p2);

#endif //FRM_C_PATTERNS_H
