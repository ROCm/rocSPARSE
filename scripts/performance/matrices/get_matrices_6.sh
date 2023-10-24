#!/usr/bin/env bash
# Author: Nico Trost

matrices=(europe_osm
          asia_osm
          germany_osm
          road_usa
          delaunay_n24
          hugebubbles-00020
          hugebubbles-00010
          hugebubbles-00000
          adaptive
          hugetric-00010
          hugetric-00000
          M6
          333SP
          AS365
          venturiLevel3
)

url=(https://sparse.tamu.edu/MM/DIMACS10
     https://sparse.tamu.edu/MM/DIMACS10
     https://sparse.tamu.edu/MM/DIMACS10
     https://sparse.tamu.edu/MM/DIMACS10
     https://sparse.tamu.edu/MM/DIMACS10
     https://sparse.tamu.edu/MM/DIMACS10
     https://sparse.tamu.edu/MM/DIMACS10
     https://sparse.tamu.edu/MM/DIMACS10
     https://sparse.tamu.edu/MM/DIMACS10
     https://sparse.tamu.edu/MM/DIMACS10
     https://sparse.tamu.edu/MM/DIMACS10
     https://sparse.tamu.edu/MM/DIMACS10
     https://sparse.tamu.edu/MM/DIMACS10
     https://sparse.tamu.edu/MM/DIMACS10
     https://sparse.tamu.edu/MM/DIMACS10
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
