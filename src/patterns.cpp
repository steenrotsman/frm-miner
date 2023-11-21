//
// Created by stijn on 6/8/23.
//

#include <vector>
#include <algorithm>

#include "patterns.h"
#include "motif.h"
#include "typing.h"

PatternMiner::PatternMiner(double minsup, int min_len, int max_len, double max_overlap) : minsup(minsup), min_len(min_len), max_len(max_len),
                                                                              max_overlap(max_overlap), min_freq(0.0),
                                                                              k(2), k_motifs(), k_1_motifs(){}

void PatternMiner::mine(const DiscreteDB& sequences)
{
    // Frequency is easier to check than support
    min_freq = static_cast<double>(sequences.size()) * minsup;

    // Mine 1-patterns separately from longer patterns
    mine_1_patterns(sequences);

    // If there were no frequent k-patterns, there can be no frequent (k+1)-patterns; stop
    for (k = 2; (!k_1_motifs.empty() and ((not max_len or k <= max_len))); k++) {
        // Generate candidate k-patterns from frequent (k-1)-patterns, find their occurrences and remove infrequent candidates
        for (auto& candidate : get_candidates()) {
            Motif motif { candidate };
            find_occurrences(candidate, sequences, motif);

            // Add frequent patterns to frequent vector
            if (static_cast<double>(motif.get_indexes().size()) >= min_freq) {
                k_motifs.insert(std::pair(candidate, motif));
                frequent.push_back(candidate);
            }
        }
        k_1_motifs = k_motifs;
        k_motifs.clear();
    }

    // Remove patterns that are too short or overlap too much with a longer pattern
    remove_redundant();
}

void PatternMiner::mine_1_patterns(const DiscreteDB& sequences)
{
    // Scan database once to record indexes of 1-patterns
    for (size_t i = 0; i < sequences.size(); i++) {
        for (size_t j = 0; j < sequences[i].size(); j++) {
            Pattern item = {sequences[i][j]};
            if (k_1_motifs.count(item) == 0) {
                k_1_motifs.insert(std::pair(item, Motif(item)));
            }
            k_1_motifs.at(item).record_index(i, j);
        }
    }

    // Add frequent patterns to frequent vector
    for (auto& [pattern, motif] : k_1_motifs) {
        if (static_cast<double>(motif.get_indexes().size()) >= min_freq) {
            frequent.push_back(pattern);
        }
    }
}

std::vector<Pattern> PatternMiner::get_candidates()
{
    std::vector<Pattern> candidates { };
    for (auto& [p1, m1] : k_1_motifs) {
        for (auto& [p2, m2] : k_1_motifs) {
            // Check if p1[1:] == p2[:-1]
            if (std::equal(p1.begin()+1, p1.end(), p2.begin(), p2.end()-1)) {
                // Construct candidate by combing p1 and p2
                Pattern candidate { p1 };
                candidate.push_back(p2.back());
                candidates.push_back(candidate);
            }
        }
    }
    return candidates;
}

void PatternMiner::find_occurrences(const Pattern& candidate, const DiscreteDB& sequences, Motif& motif)
{
    // Find candidate via its first parent
    Pattern parent(candidate.begin(), candidate.end() - 1);
    for (auto& [seq, indexes] : k_1_motifs.at(parent).get_indexes()) {
        for (auto index : indexes) {
            // If index + length is larger than sequence size, candidate can never be present at index
            // Omitting this check leads to a bug if start of next sequence would complete the pattern
            if (index + k > sequences[seq].size()) {
                continue;
            }

            Pattern possible_match(sequences[seq].begin() + index, sequences[seq].begin() + index + k);
            if (possible_match == candidate) {
                motif.record_index(seq, index);
            }
        }
    }
}

void PatternMiner::remove_redundant()
{
    // Remove patterns that are too short
    frequent.erase(
            std::remove_if(
                    frequent.begin(),
                    frequent.end(),
                    [this](const std::vector<char>& v) {
                        return v.size() < min_len;
                    }
            ),
            frequent.end()
    );

    // If max_overlap == 1, no overlap is too high
    if (max_overlap == 1) {
        return;
    }

    // Create a copy of frequent to edit frequent in the loops
    std::vector<Pattern> patterns = frequent;

    // Sort the copy in decreasing order of vector lengths
    std::sort(patterns.begin(), patterns.end(),
              [](const std::vector<char>& a, const std::vector<char>& b) {
                  return a.size() > b.size();
              });

    // Remove patterns for which more than max_overlap% is overlapping from frequent
    std::vector<Pattern> removed;
    for (const auto& p1 : patterns) {
        // Check if p1 was not already removed
        if (is_p_in_vec(p1, removed)) {
            continue;
        }
        for (const auto& p2 : patterns) {
            if (p2.size() >= p1.size() or is_p_in_vec(p2, removed)) {
                continue;
            }

            // Check if shorter patterns overlaps too much with larger pattern
            if (lcs(p1, p2) / static_cast<double>(p2.size()) > max_overlap) {
                auto pos = std::find(frequent.begin(), frequent.end(), p2);
                frequent.erase(pos);
                removed.push_back(p2);
            }
        }
    }
}

bool is_p_in_vec(const Pattern& p, const std::vector<Pattern>& vec)
{
    return std::find(vec.begin(), vec.end(), p) != vec.end();
}

double lcs(const Pattern& p1, const Pattern& p2) {
    size_t n = p1.size();
    size_t m = p2.size();

    std::vector<std::vector<int>> dp(n + 1, std::vector<int>(m + 1, 0));

    for (size_t i = 1; i <= n; i++) {
        for (size_t j = 1; j <= m; j++) {
            if (p1[i - 1] == p2[j - 1]) {
                dp[i][j] = dp[i - 1][j - 1] + 1;
            } else {
                dp[i][j] = std::max(dp[i - 1][j], dp[i][j - 1]);
            }
        }
    }

    return static_cast<double>(dp[n][m]);
}
