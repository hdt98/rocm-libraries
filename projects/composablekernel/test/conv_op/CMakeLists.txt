add_test_executable(test_conv_op conv_op.cpp)
target_link_libraries(test_conv_op PRIVATE utility)

add_test_executable(test_conv_block conv_block.cpp)
target_link_libraries(test_conv_block PRIVATE utility)

add_test_executable(test_conv_device
                    conv_device.cpp
                    conv_device_1.cpp
                    conv_device_2.cpp
                    conv_device_4.cpp
                    conv_device_8.cpp
                    conv_device_10.cpp
                    conv_device_20.cpp
                    conv_device_40.cpp
                    conv_device_80.cpp
                    conv_device_100.cpp
                    conv_device_200.cpp
                    conv_device_400.cpp)
target_link_libraries(test_conv_device PRIVATE utility)

add_test_executable(test_conv_device_wavegroup
                    conv_device.cpp
                    conv_device_1.cpp
                    conv_device_2.cpp
                    conv_device_4.cpp
                    conv_device_8.cpp
                    conv_device_10.cpp
                    conv_device_20.cpp
                    conv_device_40.cpp
                    conv_device_80.cpp
                    conv_device_100.cpp
                    conv_device_200.cpp
                    conv_device_400.cpp)
target_link_libraries(test_conv_device_wavegroup PRIVATE utility)
target_compile_definitions(test_conv_device_wavegroup PRIVATE ENABLE_WAVEGROUP=1)
target_compile_options(test_conv_device_wavegroup PRIVATE -save-temps=obj -Wno-gnu-line-marker)

add_test_executable(test_conv_device_transposed conv_device_transposed.cpp)
target_link_libraries(test_conv_device_transposed PRIVATE utility)

add_test_executable(test_conv_device_transposed_wavegroup conv_device_transposed.cpp)
target_compile_definitions(test_conv_device_transposed_wavegroup PRIVATE ENABLE_WAVEGROUP=1)
target_link_libraries(test_conv_device_transposed_wavegroup PRIVATE utility)
target_compile_options(test_conv_device_transposed_wavegroup PRIVATE -save-temps=obj -Wno-gnu-line-marker)

add_test_executable(test_conv_fma conv_fma.cpp)
target_link_libraries(test_conv_fma PRIVATE utility)
target_compile_options(test_conv_fma PRIVATE -save-temps=obj -Wno-gnu-line-marker)

add_gtest_executable(test_grouped_conv_fwd_wconv_interface grouped_conv_fwd_wconv_interface.cpp)
if(result EQUAL 0)
    target_link_libraries(test_grouped_conv_fwd_wconv_interface PRIVATE utility)
endif()
