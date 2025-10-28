
## Running Tests

### Prerequisites

1. **Build the tests**
   ```bash
   git clone https://github.com/ROCm/rocm-libraries.git
   cd rocm-libraries/projects/rocfft
   mkdir build && cd build
   cmake -DCMAKE_CXX_COMPILER=amdclang++ -DCMAKE_C_COMPILER=amdclang -DBUILD_CLIENTS=on ..
   make -j
   ```

2. **Ensure Python 3** is available (required for YAML parsing)


### Basic CTest Commands

#### Runinng Tests and sample logs
```bash
root@dell-rack-13:/dockerx/rocm-libraries/projects/rocfft/build/clients# ctest -N
Test project /dockerx/rocm-libraries/projects/rocfft/build/clients
  Test  #1: rocfft_pow2_1D
  Test  #2: rocfft_pow2_2D
  Test  #3: rocfft_pow2_3D
  Test  #4: rocfft_prime_1D
  Test  #5: rocfft_mix_1D
  Test  #6: rocfft_callback_tests
  Test  #7: rocfft_hermitian_tests
  Test  #8: rocfft_unit_test
  Test  #9: rocfft_adhoc
  Test #10: rocfft_performance
  Test #11: rocfft_emulation
  Test #12: rocfft_mpi
  Test #13: rocfft_smoke
  Test #14: rocfft_half_precision

Total Tests: 14
root@dell-rack-13:/dockerx/rocm-libraries/projects/rocfft/build/clients# ctest -L adhoc
Test project /dockerx/rocm-libraries/projects/rocfft/build/clients
    Start 9: rocfft_adhoc
1/1 Test #9: rocfft_adhoc .....................   Passed    4.23 sec

100% tests passed, 0 tests failed out of 1

Label Time Summary:
accuracy      =   4.23 sec*proc (1 test)
adhoc         =   4.23 sec*proc (1 test)
edge_cases    =   4.23 sec*proc (1 test)

Total Test time (real) =   4.23 sec
root@dell-rack-13:/dockerx/rocm-libraries/projects/rocfft/build/clients# ctest -L pow2_2D
Test project /dockerx/rocm-libraries/projects/rocfft/build/clients
    Start 2: rocfft_pow2_2D
1/1 Test #2: rocfft_pow2_2D ...................   Passed    4.19 sec

100% tests passed, 0 tests failed out of 1

Label Time Summary:
2d            =   4.19 sec*proc (1 test)
accuracy      =   4.19 sec*proc (1 test)
pow2_2D       =   4.19 sec*proc (1 test)
power_of_2    =   4.19 sec*proc (1 test)

Total Test time (real) =   4.19 sec
```

