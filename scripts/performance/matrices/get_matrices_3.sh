#!/usr/bin/env bash
# Author: Nico Trost

matrices=(ldoor
          bone010
          Rucci1
          cage15
          rajat31
          12month1
          spal_004
          crankseg_2
          torso1
)

url=(https://sparse.tamu.edu/MM/GHS_psdef
     https://sparse.tamu.edu/MM/Oberwolfach
     https://sparse.tamu.edu/MM/Rucci
     https://sparse.tamu.edu/MM/vanHeukelum
     https://sparse.tamu.edu/MM/Rajat
     https://sparse.tamu.edu/MM/Buss
     https://sparse.tamu.edu/MM/Mittelmann
     https://sparse.tamu.edu/MM/GHS_psdef
     https://sparse.tamu.edu/MM/Norris
)

for i in {0..8}; do
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
