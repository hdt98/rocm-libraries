add_test_executable(test_conv_sba_uba conv_sba_uba.cpp)
target_link_libraries(test_conv_sba_uba PRIVATE utility)

add_test_executable(test_conv_sba_uba_device
                    convsuba_device.cpp
                    convsuba_device_124.cpp)
target_link_libraries(test_conv_sba_uba_device PRIVATE utility)
