import os
import subprocess

PATH = os.path.dirname(os.path.abspath(__file__))


def main():
    files = [
        "e1_pipeline.py",
        "e2_problem_setting.py",
        "e3_cycling.py",
        "e4_insect.py",
        "e5_6_comparison.py",
        "e7_accuracy.py",
        "e8_scalability.py",
        "e9_params_analysis.py",
    ]
    for file in files:
        run(file)


def run(file):
    print(file)
    subprocess.run(["python", os.path.join(PATH, file)])


if __name__ == "__main__":
    main()
