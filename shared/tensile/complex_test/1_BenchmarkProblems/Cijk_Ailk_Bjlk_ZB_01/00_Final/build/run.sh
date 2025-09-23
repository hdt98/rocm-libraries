#!/bin/bash

set -ex
ERR=0
/home/mchirila/rocm-libraries/shared/tensile/build/client/tensile-client --config-file /home/mchirila/rocm-libraries/shared/tensile/Tensile/complex_test/1_BenchmarkProblems/Cijk_Ailk_Bjlk_ZB_01/00_Final/build/../source/ClientParameters.ini  
if [[ $? -ne 0 ]]
then
    echo error in /home/mchirila/rocm-libraries/shared/tensile/Tensile/complex_test/1_BenchmarkProblems/Cijk_Ailk_Bjlk_ZB_01/00_Final/build/../source/ClientParameters.ini
    ERR=$?
fi
exit $ERR
