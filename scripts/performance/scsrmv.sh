#!/usr/bin/env bash
# Author: Nico Trost

# Helper function
function display_help()
{
    echo "rocSPARSE benchmark helper script"
    echo "    [-h|--help] prints this help message"
    echo "    [-d|--device] select device"
    echo "    [-p|--path] path to rocsparse-bench"
}

# Check if getopt command is installed
type getopt > /dev/null
if [[ $? -ne 0 ]]; then
    echo "This script uses getopt to parse arguments; try installing the util-linux package";
    exit 1;
fi

dev=0
path=../../build/release/clients/benchmarks

# Parse command line parameters
getopt -T
if [[ $? -eq 4 ]]; then
    GETOPT_PARSE=$(getopt --name "${0}" --longoptions help,device:,path: --options hd:p: -- "$@")
else
    echo "Need a new version of getopt"
    exit 1
fi

if [[ $? -ne 0 ]]; then
    echo "getopt invocation failed; could not parse the command line";
    exit 1
fi

eval set -- "${GETOPT_PARSE}"

while true; do
    case "${1}" in
        -h|--help)
            display_help
            exit 0
            ;;
        -d|--device)
            dev=${2}
            shift 2 ;;
        -p|--path)
            path=${2}
            shift 2 ;;
        --) shift ; break ;;
        *)  echo "Unexpected command line parameter received; aborting";
            exit 1
            ;;
    esac
done

bench=$path/rocsparse-bench

# Check if binary is available
if [ ! -f $bench ]; then
    echo $bench not found, exit...
    exit 1
else
    echo ">>" $(realpath $(ldd $bench | grep rocsparse | awk '{print $3;}'))
fi

$bench -f csrmv --precision s --device $dev --alpha 1 --beta 0 --iters 200 --rocalution matrices/12month1.csr
$bench -f csrmv --precision s --device $dev --alpha 1 --beta 0 --iters 200 --rocalution matrices/ASIC_320k.csr
$bench -f csrmv --precision s --device $dev --alpha 1 --beta 0 --iters 200 --rocalution matrices/bibd_22_8.csr
$bench -f csrmv --precision s --device $dev --alpha 1 --beta 0 --iters 200 --rocalution matrices/bmw3_2.csr
$bench -f csrmv --precision s --device $dev --alpha 1 --beta 0 --iters 200 --rocalution matrices/bmw7st_1.csr
$bench -f csrmv --precision s --device $dev --alpha 1 --beta 0 --iters 200 --rocalution matrices/bmwcra_1.csr
$bench -f csrmv --precision s --device $dev --alpha 1 --beta 0 --iters 200 --rocalution matrices/bone010.csr
$bench -f csrmv --precision s --device $dev --alpha 1 --beta 0 --iters 200 --rocalution matrices/cage15.csr
$bench -f csrmv --precision s --device $dev --alpha 1 --beta 0 --iters 200 --rocalution matrices/cant.csr
$bench -f csrmv --precision s --device $dev --alpha 1 --beta 0 --iters 200 --rocalution matrices/Chebyshev4.csr
$bench -f csrmv --precision s --device $dev --alpha 1 --beta 0 --iters 200 --rocalution matrices/circuit5M.csr
$bench -f csrmv --precision s --device $dev --alpha 1 --beta 0 --iters 200 --rocalution matrices/consph.csr
$bench -f csrmv --precision s --device $dev --alpha 1 --beta 0 --iters 200 --rocalution matrices/crankseg_2.csr
$bench -f csrmv --precision s --device $dev --alpha 1 --beta 0 --iters 200 --rocalution matrices/eu-2005.csr
$bench -f csrmv --precision s --device $dev --alpha 1 --beta 0 --iters 200 --rocalution matrices/Ga41As41H72.csr
$bench -f csrmv --precision s --device $dev --alpha 1 --beta 0 --iters 200 --rocalution matrices/hood.csr
$bench -f csrmv --precision s --device $dev --alpha 1 --beta 0 --iters 200 --rocalution matrices/in-2004.csr
$bench -f csrmv --precision s --device $dev --alpha 1 --beta 0 --iters 200 --rocalution matrices/ldoor.csr
$bench -f csrmv --precision s --device $dev --alpha 1 --beta 0 --iters 200 --rocalution matrices/mac_econ_fwd500.csr
$bench -f csrmv --precision s --device $dev --alpha 1 --beta 0 --iters 200 --rocalution matrices/mc2depi.csr
$bench -f csrmv --precision s --device $dev --alpha 1 --beta 0 --iters 200 --rocalution matrices/mip1.csr
$bench -f csrmv --precision s --device $dev --alpha 1 --beta 0 --iters 200 --rocalution matrices/nd24k.csr
$bench -f csrmv --precision s --device $dev --alpha 1 --beta 0 --iters 200 --rocalution matrices/pdb1HYS.csr
$bench -f csrmv --precision s --device $dev --alpha 1 --beta 0 --iters 200 --rocalution matrices/pwtk.csr
$bench -f csrmv --precision s --device $dev --alpha 1 --beta 0 --iters 200 --rocalution matrices/rail4284.csr
$bench -f csrmv --precision s --device $dev --alpha 1 --beta 0 --iters 200 --rocalution matrices/rajat31.csr
$bench -f csrmv --precision s --device $dev --alpha 1 --beta 0 --iters 200 --rocalution matrices/rma10.csr
$bench -f csrmv --precision s --device $dev --alpha 1 --beta 0 --iters 200 --rocalution matrices/Rucci1.csr
$bench -f csrmv --precision s --device $dev --alpha 1 --beta 0 --iters 200 --rocalution matrices/s3dkq4m2.csr
$bench -f csrmv --precision s --device $dev --alpha 1 --beta 0 --iters 200 --rocalution matrices/scircuit.csr
$bench -f csrmv --precision s --device $dev --alpha 1 --beta 0 --iters 200 --rocalution matrices/shipsec1.csr
$bench -f csrmv --precision s --device $dev --alpha 1 --beta 0 --iters 200 --rocalution matrices/Si41Ge41H72.csr
$bench -f csrmv --precision s --device $dev --alpha 1 --beta 0 --iters 200 --rocalution matrices/sls.csr
$bench -f csrmv --precision s --device $dev --alpha 1 --beta 0 --iters 200 --rocalution matrices/sme3Dc.csr
$bench -f csrmv --precision s --device $dev --alpha 1 --beta 0 --iters 200 --rocalution matrices/spal_004.csr
$bench -f csrmv --precision s --device $dev --alpha 1 --beta 0 --iters 200 --rocalution matrices/torso1.csr
$bench -f csrmv --precision s --device $dev --alpha 1 --beta 0 --iters 200 --rocalution matrices/webbase-1M.csr
