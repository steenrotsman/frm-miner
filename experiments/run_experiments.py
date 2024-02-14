import os
import subprocess
from multiprocessing import Pool, cpu_count

PATH = os.path.dirname(os.path.abspath(__file__))


def main():
    files = [file for file in os.listdir(PATH) if file[0] == 'e' and file[-3:] == '.py']
    with Pool(cpu_count()) as p:
        p.map(run, files)


def run(file):
    print(file)
    subprocess.run(['python', os.path.join(PATH, file)])


if __name__ == '__main__':
    main()
