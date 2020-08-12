#!/usr/bin/env bash
DIR=/mnt/500GB/shared_directory/SYSROOT/
source /tools/Xilinx/Vivado/2018.3/settings64.sh

if [ "$(ls -A $DIR)" ]; then
    /usr/bin/time -f "Took %E to synthesize the QNet Accl." make all SYSROOT=$DIR TARGET=hw -j5
else
    echo "Root filesystem:'$DIR' is empty or not exist. Please double check."
fi