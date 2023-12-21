import warnings

import numpy as np
import pyscamp as mp
with warnings.catch_warnings():
    warnings.simplefilter("ignore")
    from mass_ts import mass2 as mass


def ostinato(ts, m):
    n = len(ts)
    bsf_rad, ts_index, ss_index = np.inf, 0, 0
    for j, _ in enumerate(ts):
        h = (j + 1) % n
        profile, _ = mp.abjoin(ts[j], ts[h], m)
        for q in np.argsort(profile):
            radius = profile[q]
            if radius >= bsf_rad:
                break
            for i, _ in enumerate(ts):
                if i not in (j, h):
                    radius = max(radius, min(mass(ts[i], ts[j][q:q+m])).real)
                    if radius >= bsf_rad:
                        break
            if radius < bsf_rad:
                bsf_rad, ts_index, ss_index = radius, j, q
    return bsf_rad, ts_index, ss_index


def k_of_p_ostinato(ts, m, k):
    n = len(ts)
    if n == k:
        return ostinato(ts, m)

    bsf_rad, ts_index, ss_index, bsf_outliers = np.inf, 0, 0, {}
    for j, _ in enumerate(ts):
        h = (j + 1) % n
        profile, _ = mp.abjoin(ts[j], ts[h], m)
        for q in np.argsort(profile):
            radius = 0
            outliers = {h: profile[q]}

            for i, _ in enumerate(ts):
                if i not in (j, h):
                    rad = min(mass(ts[i], ts[j][q:q+m])).real
                    if rad > radius:
                        outliers[i] = rad
                        if len(outliers) > n-k:
                            smallest_outlier_index, smallest_outlier_radius = min(outliers.items(), key=lambda x: x[1])
                            radius = smallest_outlier_radius
                            outliers.pop(smallest_outlier_index)
                            if radius > bsf_rad:
                                break
            if radius < bsf_rad:
                bsf_rad, ts_index, ss_index, bsf_outliers = radius, j, q, outliers
    return bsf_rad, ts_index, ss_index, bsf_outliers
