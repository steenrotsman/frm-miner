import os
import subprocess

PATH = os.path.dirname(os.path.abspath(__file__))


def main():
    files = [
        'e0_pipeline.py',
        'e1_runtime.py',
        'e1_memory.py',
        'e1_runtime_analysis.py',
        'e2_stocks.py',
        'e2_params.py',
        'e2_params_analysis.py',
        'e3_accuracy sim.py',
        'e4_scalability_sim.py',
        'e5_fit_to_json.py',
        'e5_bike.py',
        'e5_spinning.py',
        'e6_insect.py',
    ]
    for file in files:
        run(file)


def run(file):
    print(file)
    subprocess.run(['python', os.path.join(PATH, file)])


if __name__ == '__main__':
    main()
