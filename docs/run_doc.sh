#!/bin/bash

if [ -d docBin ]; then
    rm -rf docBin
fi

cur_version=$(sed -n -e "s/^.*VERSION_STRING.* \"\([0-9\.]\{1,\}\).*/\1/p" ../CMakeLists.txt)
sed -i -e "s/\(PROJECT_NUMBER.*=\)\(.*\)/\1 v${cur_version}/" Doxyfile
sed -i -e "s/\(version.*=\)\(.*\)/\1 u'${cur_version}'/" source/conf.py
sed -i -e "s/\(release.*=\)\(.*\)/\1 u'${cur_version}'/" source/conf.py

sed -e 's/ROCSPARSE_EXPORT//g' ../library/include/rocsparse-functions.h > rocsparse-functions.h
sed -e 's/ROCSPARSE_EXPORT//g' ../library/include/rocsparse-auxiliary.h > rocsparse-auxiliary.h
sed -i 's/#include "rocsparse-export.h"//g' rocsparse-functions.h
sed -i 's/#include "rocsparse-export.h"//g' rocsparse-auxiliary.h
cp ../library/include/rocsparse-types.h rocsparse-types.h
doxygen Doxyfile

cd source
make clean
make html
make latexpdf
cd ..

rm rocsparse-functions.h
rm rocsparse-auxiliary.h
rm rocsparse-types.h
