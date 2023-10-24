#!/usr/bin/env bash
# Author: Nico Trost

matrices=(rajat31
          NLR
          patents
          FullChip
          Freescale1
          Freescale2
          Hardesty3
          memchip
          CurlCurl_4
          kkt_power
          atmosmodm
          ss
          nv2
          ecology1
          ecology2
)

url=(https://sparse.tamu.edu/MM/Rajat
     https://sparse.tamu.edu/MM/DIMACS10
     https://sparse.tamu.edu/MM/Pajek
     https://sparse.tamu.edu/MM/Freescale
     https://sparse.tamu.edu/MM/Freescale
     https://sparse.tamu.edu/MM/Freescale
     https://sparse.tamu.edu/MM/Hardesty
     https://sparse.tamu.edu/MM/Freescale
     https://sparse.tamu.edu/MM/Bodendiek
     https://sparse.tamu.edu/MM/Zaoui
     https://sparse.tamu.edu/MM/Bourchtein
     https://sparse.tamu.edu/MM/VLSI
     https://sparse.tamu.edu/MM/VLSI
     https://sparse.tamu.edu/MM/McRae
     https://sparse.tamu.edu/MM/McRae
)

for i in {0..14}; do
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
