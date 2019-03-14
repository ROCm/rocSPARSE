#!/usr/bin/env bash
# Author: Nico Trost

matrices=(bibd_22_8
          sls
          Chebyshev4
          sme3Dc
          ASIC_320k
          nd24k
          bmw3_2
          hood
          bmwcra_1
          bmw7st_1
          s3dkq4m2
)

url=(https://sparse.tamu.edu/MM/JGD_BIBD
     https://sparse.tamu.edu/MM/Bates
     https://sparse.tamu.edu/MM/Muite
     https://sparse.tamu.edu/MM/FEMLAB
     https://sparse.tamu.edu/MM/Sandia
     https://sparse.tamu.edu/MM/ND
     https://sparse.tamu.edu/MM/GHS_indef
     https://sparse.tamu.edu/MM/GHS_psdef
     https://sparse.tamu.edu/MM/GHS_psdef
     https://sparse.tamu.edu/MM/GHS_psdef
     https://sparse.tamu.edu/MM/GHS_psdef
)

for i in {0..10}; do
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
