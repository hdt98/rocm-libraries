#!/usr/bin/cmake -P
# Copyright Advanced Micro Devices, Inc., or its affiliates.
# SPDX-License-Identifier: MIT

find_program(ROCMINFO_EXECUTABLE rocminfo)

if(NOT ROCMINFO_EXECUTABLE)
    message(FATAL_ERROR "rocminfo not found")
endif()

execute_process(
    COMMAND ${ROCMINFO_EXECUTABLE}
    RESULT_VARIABLE ROCMINFO_EXIT_CODE
    OUTPUT_VARIABLE ROCMINFO_STDOUT
    ERROR_VARIABLE ROCMINFO_STDERR
)

if(ROCMINFO_EXIT_CODE)
    message(SEND_ERROR "rocminfo exited with ${ROCMINFO_EXIT_CODE}")
    message(SEND_ERROR ${ROCMINFO_STDOUT})
    message(FATAL_ERROR ${ROCMINFO_STDERR})
endif()

string(REGEX MATCHALL [[--(gfx[0-9a-f]+)]]
    ROCMINFO_MATCHES
    ${ROCMINFO_STDOUT}
)

set(GFXIP_AND_ID)
set(ID 0)
foreach(ROCMINFO_MATCH IN LISTS ROCMINFO_MATCHES)
    string(REGEX REPLACE "--" "" ROCMINFO_MATCH ${ROCMINFO_MATCH})
    list(APPEND GFXIP_AND_ID "${ROCMINFO_MATCH}:${ID}")
    math(EXPR ID "${ID} + 1")
endforeach()
list(SORT GFXIP_AND_ID)

set(JSON_PAYLOAD)
set(IT1 0)
list(GET GFXIP_AND_ID ${IT1} I1)
string(REGEX REPLACE ":[0-9a-f]+" "" IP1 ${I1})
list(LENGTH GFXIP_AND_ID COUNT)
while(IT1 LESS COUNT)
    string(APPEND JSON_PAYLOAD "\n      \"${IP1}\": [")
    set(IT2 ${IT1})
    list(GET GFXIP_AND_ID ${IT2} I2)
    string(REGEX REPLACE [[:[0-9a-f]+$]] "" IP2 ${I2})
    string(REGEX REPLACE [[^gfx[0-9a-f]+:]] "" ID2 ${I2})
    while(${IP2} STREQUAL ${IP1} AND IT2 LESS COUNT)
        string(APPEND JSON_PAYLOAD
            "\n        {\n"
            "           \"id\": \"${ID2}\"\n"
            "        },"
        )
        math(EXPR IT2 "${IT2} + 1")
        if(IT2 LESS COUNT)
            list(GET GFXIP_AND_ID ${IT2} I2)
            string(REGEX REPLACE [[:[0-9a-f]+$]] "" IP2 ${I2})
            string(REGEX REPLACE [[^gfx[0-9a-f]+:]] "" ID2 ${I2})
        endif()
    endwhile()
    string(REGEX REPLACE [[,$]] "" JSON_PAYLOAD ${JSON_PAYLOAD})
    string(APPEND JSON_PAYLOAD "\n      ],")
    set(IT1 ${IT2})
    set(IP1 ${IP2})
endwhile()
string(REGEX REPLACE [[,$]] "" JSON_PAYLOAD ${JSON_PAYLOAD})

set(JSON_HEAD [[{
  "version": {
    "major": 1,
    "minor": 0
  },
  "local": [
    {]]
)
set(JSON_TAIL [[

    }
  ]
}]]
)

file(WRITE ${CMAKE_CURRENT_BINARY_DIR}/resources.json
    ${JSON_HEAD}
    ${JSON_PAYLOAD}
    ${JSON_TAIL}
)
