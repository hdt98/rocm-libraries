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