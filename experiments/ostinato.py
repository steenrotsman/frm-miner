import warnings

import numpy as np
import pyscamp as mp
with warnings.catch_warnings():
    warnings.simplefilter("ignore")
    from mass_ts import mass2 as mass


def ostinato(ts, m):
    k = len(ts) - 1
    bsf_rad, ts_index, ss_index = np.inf, 0, 0
    for j, _ in enumerate(ts):
        h = j + 1 if j < k else 0
        profile, _ = mp.abjoin(ts[j], ts[h], m)
        for q in np.argsort(profile):
            radius = profile[q]
            if radius >= bsf_rad:
                break
            for i, _ in enumerate(ts):
                if i not in (j, h):
                    radius = max(radius, min(mass(ts[i], ts[j][q:q+m])))
                    if radius >= bsf_rad:
                        break
            if radius < bsf_rad:
                bsf_rad, ts_index, ss_index = radius, j, q
    return bsf_rad, ts_index, ss_index