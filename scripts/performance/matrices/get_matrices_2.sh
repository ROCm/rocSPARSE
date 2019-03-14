#!/usr/bin/env bash
# Author: Nico Trost

matrices=(circuit5M
          eu-2005
          Ga41As41H72
          in-2004
          mip1
          Si41Ge41H72)

url=(https://sparse.tamu.edu/MM/Freescale
     https://sparse.tamu.edu/MM/LAW
     https://sparse.tamu.edu/MM/PARSEC
     https://sparse.tamu.edu/MM/LAW
     https://sparse.tamu.edu/MM/Andrianov
     https://sparse.tamu.edu/MM/PARSEC)

for i in {0..5}; do
    m=${matrices[${i}]}
    u=${url[${i}]}
    if [ ! -f ${m}.csr ]; then
        if [ ! -f ${m}.mtx ]; then
            if [ ! -f ${m}.tar.gz ]; then
                echo "Downloading ${m}.tar.gz ..."
                wget ${u}/${m}.tar.gz
            fi
            echo "Extracting ${m}.tar.gz ..."
            tar xf ${m}.tar.gz && mv ${m}/${m}.mtx . && rm -rf ${m}.tar.gz ${m}
        fi
        echo "Converting ${m}.mtx ..."
        ./convert ${m}.mtx ${m}.csr
        rm ${m}.mtx
    fi
done
