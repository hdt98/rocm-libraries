add_library(MIOpen_shared STATIC
    env.cpp
)

set_target_properties(MIOpen_static PROPERTIES POSITION_INDEPENDENT_CODE ON)