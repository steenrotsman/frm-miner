from collections import defaultdict

FILE = 'e1_runtime.csv'

runtimes = defaultdict(float)
with open(FILE) as fp:
    for line in fp.readlines():
        row = line[:-1].split(',')
        runtimes[row[0]] += float(row[2])

for k, v in runtimes.items():
    print(f'{k[10:]}: {v/3600:.2f}')
