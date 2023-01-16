#!/usr/bin/env bash
# Author: Yvan Mokwinski

mkdir build && cd build && cmake .. && make && cp ./convert .. && cd .. && rm -r build
