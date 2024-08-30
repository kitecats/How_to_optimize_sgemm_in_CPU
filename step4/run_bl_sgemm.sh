#!/bin/bash

#Single Thread
k_start=12
k_end=1024
k_blocksize=12
echo "result=["
echo -e "%m\t%n\t%k\t%MY_MFLOPS\t%REF_MFLOPS"
for (( k=k_start; k<=k_end; k+=k_blocksize ))
do
    ./test_bl_sgemm_step4.x     $k $k $k 
done
echo "];"

