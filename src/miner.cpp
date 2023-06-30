//
// Created by stijn on 6/8/23.
//
#include <map>
#include <vector>
#include <algorithm>
#include <iostream>

#include "typing.h"
#include "miner.h"
#include "sax.h"
#include "patterns.h"
#include "motif.h"

Miner::Miner(double minsup, int seglen, int alphabet, int min_len, double max_overlap, int k) : minsup(minsup), seglen(seglen), alphabet(alphabet), min_len(min_len), max_overlap(max_overlap), k(k){}

std::vector<Motif> Miner::mine(TimeSeriesDB& timeseries)
{
    znorm(timeseries);
    DiscreteDB sequences { sax(timeseries, seglen, alphabet) };
    mine_patterns(sequences);
    map_patterns(timeseries);
    sort_patterns();

    if (k == 0) {
        return motifs;
    } else {
        // Return k motifs or less if motifs size is less than k
        return std::vector<Motif>(motifs.begin(), motifs.begin() + std::min(k, static_cast<int>(motifs.size())));
    }
}

void Miner::mine_patterns(DiscreteDB& sequences)
{
    PatternMiner pm { minsup, min_len, max_overlap };
    pm.mine(sequences);
    for (auto& [pattern, motif] : pm.get_frequent()) {
        motifs.push_back(motif);
    }
}

void Miner::map_patterns(TimeSeriesDB& timeseries)
{
    for (auto& motif : motifs) {
        motif.map(timeseries, seglen);
    }
}

void Miner::sort_patterns()
{
    // Sort the motifs in increasing order based on naed attribute
    std::sort(motifs.begin(), motifs.end(), [](Motif& m1, Motif& m2) {
        return m1.get_naed() < m2.get_naed();
    });
}
