//
// Created by stijn on 6/8/23.
//

#ifndef FRM_C_MINER_H
#define FRM_C_MINER_H

#include <vector>

#include "typing.h"
#include "motif.h"

class Miner {
private:
    double minsup;
    int seglen;
    int alphabet;
    int min_len;
    double max_overlap;
    int k;
    std::vector<Motif> motifs;

    void mine_patterns(DiscreteDB& sequences);
    void map_patterns(TimeSeriesDB& timeseries);
    void sort_patterns();
public:
    Miner(double minsup, int seglen, int alphabet, int min_len=3, double max_overlap=0.9, int k=0);
    std::vector<Motif> mine(TimeSeriesDB& timeseries);
    std::vector<Motif> get_motifs() { return motifs; };
};


#endif //FRM_C_MINER_H
