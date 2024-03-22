import os
import subprocess

PATH = os.path.dirname(os.path.abspath(__file__))


def main():
    files = [file for file in os.listdir(PATH) if file[0] == 'e' and file[-3:] == '.py']
    for file in files:
        run(file)


def run(file):
    print(file)
    subprocess.run(['python', os.path.join(PATH, file)])


if __name__ == '__main__':
    main()
