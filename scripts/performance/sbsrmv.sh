#!/usr/bin/env bash
# Author: Nico Trost

# Helper function
function display_help()
{
    echo "rocSPARSE benchmark helper script"
    echo "    [-h|--help] prints this help message"
    echo "    [-d|--device] select device"
    echo "    [-p|--path] path to rocsparse-bench"
    echo "    [-b|--blockdim] BSR block dimension"
}

# Check if getopt command is installed
type getopt > /dev/null
if [[ $? -ne 0 ]]; then
    echo "This script uses getopt to parse arguments; try installing the util-linux package";
    exit 1;
fi

blockdim=3
dev=0
path=../../build/release/clients/staging

# Parse command line parameters
getopt -T
if [[ $? -eq 4 ]]; then
    GETOPT_PARSE=$(getopt --name "${0}" --longoptions help,device:,path:,blockdim: --options hd:p:b: -- "$@")
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
        -b|--blockdim)
            blockdim=${2}
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

# Generate logfile name
logname=sbsrmv_$(date +'%Y%m%d%H%M%S').log
truncate -s 0 $logname

# Run bsrmv for all matrices available
for filename in ./matrices/*.csr; do
    $bench -f bsrmv --precision s --device $dev --blockdim $blockdim --alpha 1 --beta 0 --iters 1000 --rocalution $filename 2>&1 | tee -a $logname
done
