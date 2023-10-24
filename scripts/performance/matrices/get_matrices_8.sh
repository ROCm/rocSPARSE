#!/usr/bin/env bash
# Author: Nico Trost

matrices=(circuit5M
          com-Orkut
          kron_g500-logn21
          kron_g500-logn20
          kron_g500-logn19
          bundle_adj
          SiO2
          mycielskian16
          mycielskian15
          mouse_gene
          av41092
          Maragal_8
          human_gene1
          human_gene2
          std1_Jac3
)

url=(https://sparse.tamu.edu/MM/Freescale
     https://sparse.tamu.edu/MM/SNAP
     https://sparse.tamu.edu/MM/DIMACS10
     https://sparse.tamu.edu/MM/DIMACS10
     https://sparse.tamu.edu/MM/DIMACS10
     https://sparse.tamu.edu/MM/Mazaheri
     https://sparse.tamu.edu/MM/PARSEC
     https://sparse.tamu.edu/MM/Mycielski
     https://sparse.tamu.edu/MM/Mycielski
     https://sparse.tamu.edu/MM/Belcastro
     https://sparse.tamu.edu/MM/Vavasis
     https://sparse.tamu.edu/MM/NYPA
     https://sparse.tamu.edu/MM/Belcastro
     https://sparse.tamu.edu/MM/Belcastro
     https://sparse.tamu.edu/MM/VanVelzen
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
