#!/bin/bash
rm  *.so 
python setup.py build_ext --inplace
rm -rf ./build


