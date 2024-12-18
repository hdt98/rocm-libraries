add_test_executable(test_conv_sba_uba conv_sba_uba.cpp)
target_link_libraries(test_conv_sba_uba PRIVATE utility)

add_test_executable(test_conv_sba_uba_device
                    convsuba_device.cpp
                    conv_device_00_1.cpp
                    conv_device_00_2.cpp
                    conv_device_00_4.cpp
                    conv_device_01_1.cpp
                    conv_device_01_2.cpp
                    conv_device_01_4.cpp
                    conv_device_10_1.cpp
                    conv_device_10_2.cpp
                    conv_device_10_4.cpp)
target_link_libraries(test_conv_sba_uba_device PRIVATE utility)

add_test_executable(test_conv_device_fma conv_device_fma.cpp)
target_link_libraries(test_conv_device_fma PRIVATE utility)