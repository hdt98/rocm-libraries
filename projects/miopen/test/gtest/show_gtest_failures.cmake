# show_gtest_failures.cmake
#
# Parses GTest JUnit XML files written by each shard and prints a single
# consolidated list of every failing test case across all shards.
#
# Called automatically by CTest via FIXTURES_CLEANUP after all shards finish,
# regardless of whether any shard passed or failed.
#
# Required variables (passed via -D on the cmake -P command line):
#   SHARD_DIR            - directory containing the per-shard XML files
#   TEST_NAME            - base name of the gtest binary (e.g. miopen_gtest)
#   GTEST_PARALLEL_LEVEL - number of shards

file(GLOB xml_files "${SHARD_DIR}/${TEST_NAME}_shard*.xml")

set(all_failures "")

foreach(xml ${xml_files})
    if(NOT EXISTS "${xml}")
        continue()
    endif()

    file(STRINGS "${xml}" lines)

    set(current_test "")
    set(in_failure FALSE)

    foreach(line ${lines})
        # Capture the test name whenever we enter a <testcase> element.
        if(line MATCHES "<testcase ")
            string(REGEX MATCH "classname=\"([^\"]*)\"" _ "${line}")
            set(cn "${CMAKE_MATCH_1}")
            string(REGEX MATCH " name=\"([^\"]*)\"" _ "${line}")
            set(nm "${CMAKE_MATCH_1}")
            set(current_test "${cn}.${nm}")
            set(in_failure FALSE)
        endif()

        # A <failure> child element means this test case failed.
        if(line MATCHES "<failure")
            set(in_failure TRUE)
        endif()

        # Record the test name when we close the element.
        if(in_failure AND line MATCHES "</testcase>")
            list(APPEND all_failures "${current_test}")
            set(in_failure FALSE)
        endif()
    endforeach()
endforeach()

list(REMOVE_DUPLICATES all_failures)
list(SORT all_failures)

message("\n========================================================")
message("CONSOLIDATED GTEST FAILURES ACROSS ALL SHARDS:")
message("========================================================")
if(all_failures)
    foreach(f ${all_failures})
        message("  FAILED: ${f}")
    endforeach()
else()
    message("  (none — all shards passed)")
endif()
message("========================================================\n")
