from random import randint, seed

from numpy import linalg

from frm.patterns import PatternMiner
from frm.preprocessing import sax, standardise

seed(0)

ts = [[randint(0, 99) / 10 for i in range(8)] for j in range(4)]
norm = standardise(ts)
seq = sax(norm, 2, 3)
amp = ' & '
end = ' \\\\\n'
for row in ts:
    print(amp.join(str(num) for num in row), end=end)
print('\midrule')
for row in norm:
    print(amp.join(str(round(num, 3)) for num in row), end=end)
print('\midrule')
for row in seq:
    print(amp.join('\multicolumn{2}{c}{' + char + '}' for char in row), end=end)

pm = PatternMiner(0.75, 1)
pm.mine(seq)
print(list(pm.frequent.keys()))
motif = pm.frequent['bb']
print(motif.get_all_indexes())

motif.map(norm, 2, 2)
for row in motif.average_occurrences.values():
    print(amp.join('\multirow{2}*{' + str(round(num, 3)) + '}' for num in row), end=end)
print('{' + ', '.join(str(round(num, 3)) for num in motif.representative) + '}')

for i, indexes in motif.get_all_indexes().items():
    for index in indexes:
        occurrence = motif.get_occurrence(norm[i], index)
        dist = linalg.norm(occurrence - motif.representative) / motif.length ** (1 / 2)
        print(round(dist, 3))
print()
print(round(motif.distance, 3))
