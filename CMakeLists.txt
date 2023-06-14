cmake_minimum_required(VERSION 3.24)
project(frm_miner)

set(CMAKE_CXX_STANDARD 14)

include_directories(src)

add_executable(frm_miner
        src/main.cpp
        src/miner.cpp
        src/miner.h
        src/motif.cpp
        src/motif.h
        src/patterns.cpp
        src/patterns.h
        src/sax.cpp
        src/sax.h
        src/typing.h)
