import warnings
from multiprocessing import Pool
from time import perf_counter

import numpy as np
import pyscamp as mp

with warnings.catch_warnings():
    warnings.simplefilter("ignore")
    from mass_ts import mass2 as mass

from e1_runtime import FILES, get_data, get_length

FILE = 'e1_profile.csv'


def main():
    # Get already calculated combinations from file
    with open(FILE) as fp:
        seen = [row.split(',')[0] for row in fp.readlines()]

    with Pool(processes=16, maxtasksperchild=1) as p:
        unseen = [n for n in FILES if n not in seen]
        p.map(benchmark, unseen)


def benchmark(name):
    print(f'{name}...')
    data = get_data(name)
    join_time, mass_time = profile_ostinato(*get_length(data))
    with open(FILE, 'a') as fp:
        fp.write(f'{name},{join_time},{mass_time}\n')
    print(f'{name} done!')


def profile_ostinato(ts, m):
    join_time = 0
    mass_time = 0
    k = len(ts) - 1
    bsf_rad = np.inf
    for j, _ in enumerate(ts):
        h = j + 1 if j < k else 0
        start = perf_counter()
        profile, _ = mp.abjoin(ts[j], ts[h], m)
        join_time += perf_counter() - start
        for q in np.argsort(profile):
            radius = profile[q]
            if radius >= bsf_rad:
                break
            for i, _ in enumerate(ts):
                if i not in (j, h):
                    start = perf_counter()
                    dists = mass(ts[i], ts[j][q : q + m])
                    mass_time += perf_counter() - start
                    radius = max(radius, min(dists))
                    if radius >= bsf_rad:
                        break
            if radius < bsf_rad:
                bsf_rad, _, _ = radius, j, q
    return join_time, mass_time


if __name__ == '__main__':
    main()
