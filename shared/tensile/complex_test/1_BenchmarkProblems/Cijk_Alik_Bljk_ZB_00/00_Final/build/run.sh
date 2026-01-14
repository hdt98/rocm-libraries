#!/bin/bash

set -ex
ERR=0
<<<<<<< HEAD
<<<<<<< HEAD
=======
>>>>>>> 20f15b7814 (Fixed validator bug by removing conjugate code)
/home/mchirila/rocm-libraries/shared/tensile/build/client/tensile-client --config-file /home/mchirila/rocm-libraries/shared/tensile/complex_test/1_BenchmarkProblems/Cijk_Alik_Bljk_ZB_00/00_Final/build/../source/ClientParameters.ini  
if [[ $? -ne 0 ]]
then
    echo error in /home/mchirila/rocm-libraries/shared/tensile/complex_test/1_BenchmarkProblems/Cijk_Alik_Bljk_ZB_00/00_Final/build/../source/ClientParameters.ini
<<<<<<< HEAD
=======
/home/mchirila/rocm-libraries/shared/tensile/build/client/tensile-client --config-file /home/mchirila/rocm-libraries/shared/tensile/Tensile/complex_test/1_BenchmarkProblems/Cijk_Alik_Bljk_ZB_00/00_Final/build/../source/ClientParameters.ini  
if [[ $? -ne 0 ]]
then
    echo error in /home/mchirila/rocm-libraries/shared/tensile/Tensile/complex_test/1_BenchmarkProblems/Cijk_Alik_Bljk_ZB_00/00_Final/build/../source/ClientParameters.ini
>>>>>>> ef5d8c342a (Changed complex test folder name for consistency, and added the files to commit.)
=======
>>>>>>> 20f15b7814 (Fixed validator bug by removing conjugate code)
    ERR=$?
fi
exit $ERR
