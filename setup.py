from pybind11.setup_helpers import Pybind11Extension, build_ext
from setuptools import setup

__version__ = "0.0.1"

ext_modules = [
    Pybind11Extension("_frm_cpp",
        ["src/main.cpp", "src/miner.cpp", "src/motif.cpp", "src/patterns.cpp", "src/sax.cpp"],
        define_macros = [('VERSION_INFO', __version__)],
        ),
]

setup(
    name="frm",
    version=__version__,
    author="Stijn J. Rotman",
    author_email="s.j.rotman@tilburguniversity.edu",
    url="https://github.com/steenrotsman/frm-miner",
    description="Mine Frequent Representative Motifs",
    long_description="",
    ext_modules=ext_modules,
    extras_require={"test": "pytest"},
    cmdclass={"build_ext": build_ext},
    zip_safe=False,
    python_requires=">=3.9",
)
