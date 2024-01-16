from pybind11.setup_helpers import Pybind11Extension, build_ext
from setuptools import setup

__version__ = "0.0.7"

ext_modules = [
    Pybind11Extension("_frm_cpp",
                      ["src/bind.cpp", "src/miner.cpp", "src/motif.cpp", "src/patterns.cpp", "src/sax.cpp"],
                      define_macros=[('VERSION_INFO', __version__)],
                      ),
]

setup(
    name="frm",
    ext_modules=ext_modules,
    cmdclass={"build_ext": build_ext},
)
