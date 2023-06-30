//
// Created by stijn on 6/8/23.
//

#include <vector>
#include <algorithm>

#include "patterns.h"
#include "motif.h"
#include "typing.h"

PatternMiner::PatternMiner(double minsup, long min_len, double max_overlap) : minsup(minsup), min_len(min_len),
                                                                              max_overlap(max_overlap), min_freq(0.0),
                                                                              k(2), patterns({{}, {}}){}

void PatternMiner::mine(const DiscreteDB& sequences)
{
    // Frequency is easier to check than support
    min_freq = static_cast<double>(sequences.size()) * minsup;

    // Mine 1-patterns separately from longer patterns
    mine_1_patterns(sequences);

    // If there were no frequent k-patterns, there can be no frequent (k+1)-patterns; stop
    for (k = 2; !patterns[k - 1].empty(); k++) {
        patterns.emplace_back();

        // Generate candidate k-patterns from frequent (k-1)-patterns, find their occurrences and remove infrequent candidates
        for (auto& candidate : get_candidates()) {
            Motif motif { candidate };
            frequent.insert(std::make_pair(candidate, motif));
            find_candidate(candidate, sequences);
            prune_infrequent(candidate);
        }
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
            if (frequent.count(item) == 0) {
                frequent.insert(std::pair(item, Motif(item)));
            }
            frequent.at(item).record_index(i, j);
        }
    }


    // Copy keys of frequent to not alter the map in the loop
    std::vector<Pattern> candidates;
    for (const auto& pair : frequent) {
        candidates.push_back(pair.first);
    }

    // Prune infrequent patterns
    for (auto& candidate : candidates) {
        prune_infrequent(candidate);
    }
}

void PatternMiner::prune_infrequent(const Pattern& pattern)
{
    if (static_cast<double>(frequent.at(pattern).get_indexes().size()) < min_freq) {
            frequent.erase(pattern);
    } else {
        patterns[pattern.size()].push_back(pattern);
    }
}

std::vector<Pattern> PatternMiner::get_candidates()
{
    const std::vector<Pattern>& prev_patterns { patterns[k - 1] };
    std::vector<Pattern> candidates { };
    for (auto& p1 : prev_patterns) {
        for (auto& p2 : prev_patterns) {
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

void PatternMiner::find_candidate(const Pattern& candidate, const DiscreteDB& sequences)
{
    // Find candidate via its first parent
    Pattern parent(candidate.begin(), candidate.end() - 1);
    for (auto& [seq, indexes] : frequent.at(parent).get_indexes()) {
        for (auto index : indexes) {
            // If index + length is larger than sequence size, candidate can never be present at index
            // Omitting this check leads to a bug if start of next sequence would complete the pattern
            if (index + k > sequences[seq].size()) {
                continue;
            }

            Pattern possible_match(sequences[seq].begin() + index, sequences[seq].begin() + index + k);
            if (possible_match == candidate) {
                frequent.at(candidate).record_index(seq, index);
            }
        }
    }
}

void PatternMiner::remove_redundant()
{
    auto flat_patterns { remove_short() };

    std::vector<Pattern> removed;
    for (const auto& p1 : flat_patterns) {
        // Check if p1 was not already removed
        if (is_p_in_vec(p1, removed)) {
            continue;
        }
        for (const auto& p2 : flat_patterns) {
            if (p2.size() > p1.size() or p1 == p2 or is_p_in_vec(p2, removed)) {
                continue;
            }

            // Check if shorter patterns overlaps too much with larger pattern
            if (lcs(p1, p2) / static_cast<double>(p2.size()) > max_overlap) {
                frequent.erase(p2);
                removed.push_back(p2);
            }
        }
    }
}

std::vector<Pattern> PatternMiner::remove_short()
{
    // Flatten patterns in increasing length, don't add too short patterns
    std::vector<Pattern> flat_patterns;
    for (int i { k-2 }; i > 0; i--) {
        for (const auto& pattern : patterns[i]) {
            if (i < min_len) {
                // Remove too short patterns
                frequent.erase(pattern);
            } else {
                // Flatten vector of long enough patterns
                flat_patterns.push_back(pattern);
            }
        }
    }
    return flat_patterns;
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
