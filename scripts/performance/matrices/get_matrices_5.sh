#!/usr/bin/env bash
# Author: Nico Trost

matrices=(mesh1e1
          bcspwr03
          can_96
          pde225
          dwt_245
          oscil_dcop_17
          tols340
          str_200
          fs_541_1
          steam2
          jagmesh1
          poli
          D_9
          ex33
          c-24
          shermanACd
          fv2
          bcsstk24
          nasa2910
          jan99jac100
          rajat22
          torsion1
          copter1
          man_5976
          benzene
          s1rmq4m1
          epb3
          TF17
          bundle1
          ss1
          sx-superuser
          crystk02
          ex11
          xenon1
)

url=(https://sparse.tamu.edu/MM/Pothen
     https://sparse.tamu.edu/MM/HB
     https://sparse.tamu.edu/MM/HB
     https://sparse.tamu.edu/MM/Bai
     https://sparse.tamu.edu/MM/HB
     https://sparse.tamu.edu/MM/Sandia
     https://sparse.tamu.edu/MM/Bai
     https://sparse.tamu.edu/MM/HB
     https://sparse.tamu.edu/MM/HB
     https://sparse.tamu.edu/MM/HB
     https://sparse.tamu.edu/MM/HB
     https://sparse.tamu.edu/MM/Grund
     https://sparse.tamu.edu/MM/JGD_SL6
     https://sparse.tamu.edu/MM/FIDAP
     https://sparse.tamu.edu/MM/Schenk_IBMNA
     https://sparse.tamu.edu/MM/Shen
     https://sparse.tamu.edu/MM/Norris
     https://sparse.tamu.edu/MM/HB
     https://sparse.tamu.edu/MM/HB
     https://sparse.tamu.edu/MM/Hollinger
     https://sparse.tamu.edu/MM/Rajat
     https://sparse.tamu.edu/MM/GHS_psdef
     https://sparse.tamu.edu/MM/GHS_psdef
     https://sparse.tamu.edu/MM/HB
     https://sparse.tamu.edu/MM/PARSEC
     https://sparse.tamu.edu/MM/Cylshell
     https://sparse.tamu.edu/MM/Averous
     https://sparse.tamu.edu/MM/JGD_Forest
     https://sparse.tamu.edu/MM/Lourakis
     https://sparse.tamu.edu/MM/VLSI
     https://sparse.tamu.edu/MM/SNAP
     https://sparse.tamu.edu/MM/Boeing
     https://sparse.tamu.edu/MM/FIDAP
     https://sparse.tamu.edu/MM/Ronis
)

for i in {0..33}; do
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
