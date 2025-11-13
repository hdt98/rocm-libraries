# Copyright Advanced Micro Devices, Inc., or its affiliates.
# SPDX-License-Identifier: MIT

include(GNUInstallDirs)

rocm_add_sphinx_doc(
    "${CMAKE_CURRENT_SOURCE_DIR}"
    BUILDER html
    OUTPUT_DIR html
    USES_DOXYGEN
)

install(
    DIRECTORY "${CMAKE_CURRENT_BINARY_DIR}/html"
    DESTINATION ${CMAKE_INSTALL_DOCDIR}
)
