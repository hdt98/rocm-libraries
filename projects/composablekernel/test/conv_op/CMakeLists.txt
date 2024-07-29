add_test_executable(test_conv_op conv_op.cpp)
target_link_libraries(test_conv_op PRIVATE utility)

add_test_executable(test_conv_block conv_block.cpp)
target_link_libraries(test_conv_block PRIVATE utility)
