# show_gtest_failures.cmake
#
# Parses two sources of test result data and writes failures.log:
#
#   1. Per-shard GTest JUnit XML (miopen_gtest_shard*.xml)
#      Written by --gtest_output=xml.  The <failure message=".."> attribute
#      holds the full assertion text (file:line + values), encoded with XML
#      character entities.  This gives individual GTest-case granularity.
#
#   2. CTest LastTest.log (Testing/Temporary/LastTest.log)
#      Written progressively by CTest as each test completes — available
#      during FIXTURES_CLEANUP unlike ctest --output-junit which is written
#      only when ctest exits.  Shard entries are skipped here because they
#      are already covered by source 1 above.
#
# Called automatically by CTest via FIXTURES_CLEANUP after all shards finish,
# regardless of whether any shard passed or failed.
#
# Required variables (passed via -D on the cmake -P command line):
#   SHARD_DIR            - directory containing the per-shard GTest XML files
#   TEST_NAME            - base name of the gtest binary (e.g. miopen_gtest)
#   GTEST_PARALLEL_LEVEL - number of shards
#
# The CTest LastTest.log is expected at:
#   <SHARD_DIR>/../../Testing/Temporary/LastTest.log

# ── Helper: decode XML attribute entities from GTest's EscapeXml() ───────────
macro(decode_xml_entities var)
    # Order matters: &amp; must come last to avoid double-decoding.
    string(REPLACE "&#x0A;" "\n"  ${var} "${${var}}")
    string(REPLACE "&#x0D;" "\r"  ${var} "${${var}}")
    string(REPLACE "&#x09;" "\t"  ${var} "${${var}}")
    string(REPLACE "&lt;"   "<"   ${var} "${${var}}")
    string(REPLACE "&gt;"   ">"   ${var} "${${var}}")
    string(REPLACE "&quot;" "\""  ${var} "${${var}}")
    string(REPLACE "&apos;" "'"   ${var} "${${var}}")
    string(REPLACE "&amp;"  "&"   ${var} "${${var}}")
endmacro()

# ━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━
# Part 1 — GTest per-shard XML (individual test-case granularity)
# ━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━

file(GLOB shard_xml_files "${SHARD_DIR}/${TEST_NAME}_shard*.xml")
list(SORT shard_xml_files)

set(gtest_failures "")   # list of "classname.testname" strings
set(gtest_log "")        # log content for gtest section

foreach(xml ${shard_xml_files})
    if(NOT EXISTS "${xml}")
        continue()
    endif()

    get_filename_component(xml_stem "${xml}" NAME_WE)
    string(REGEX REPLACE ".*_(shard[0-9]+)$" "\\1" shard_label "${xml_stem}")

    file(STRINGS "${xml}" lines)

    set(current_test "")
    set(in_failure FALSE)
    set(failure_msg "")

    foreach(line ${lines})
        if(line MATCHES "<testcase ")
            string(REGEX MATCH "classname=\"([^\"]*)\"" _ "${line}")
            set(cn "${CMAKE_MATCH_1}")
            string(REGEX MATCH " name=\"([^\"]*)\"" _ "${line}")
            set(nm "${CMAKE_MATCH_1}")
            set(current_test "${cn}.${nm}")
            set(in_failure FALSE)
            set(failure_msg "")
        endif()

        if(line MATCHES "<failure ")
            set(in_failure TRUE)
            string(REGEX MATCH "message=\"([^\"]*)\"" _ "${line}")
            set(failure_msg "${CMAKE_MATCH_1}")
            decode_xml_entities(failure_msg)
        endif()

        if(in_failure AND line MATCHES "</testcase>")
            list(APPEND gtest_failures "${current_test}")
            string(APPEND gtest_log
                "==== [${shard_label}] FAILED: ${current_test} ====\n"
                "${failure_msg}\n\n")
            set(in_failure FALSE)
            set(failure_msg "")
        endif()
    endforeach()
endforeach()

# ━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━
# Part 2 — CTest LastTest.log (non-shard CTest tests)
# ━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━
#
# ctest --output-junit writes its XML only when ctest exits, so it is never
# available when FIXTURES_CLEANUP runs.  Testing/Temporary/LastTest.log is
# written progressively by CTest as each test completes, so it IS available.
#
# Format of LastTest.log (CTest appends one block per test):
#   ----------------------------------------------------------
#   <N>/<TOTAL> Testing: <test-name>
#   ...
#   Output:
#   ----------------------------------------------------------
#   <captured output>
#   ----------------------------------------------------------
#   <N>/<TOTAL> Test: <test-name> ........ Passed/Failed <time>

# SHARD_DIR is .../build/test/gtest; one DIRECTORY hop up = .../build/test,
# which is the directory from which ctest runs (where LastTest.log is written).
get_filename_component(ctest_binary_dir "${SHARD_DIR}" DIRECTORY)
set(last_test_log "${ctest_binary_dir}/Testing/Temporary/LastTest.log")

set(ctest_failures "")   # list of test names
set(ctest_log "")        # log content for ctest section

if(EXISTS "${last_test_log}")
    file(STRINGS "${last_test_log}" lines)

    set(current_test "")
    set(capturing FALSE)
    set(output_lines "")

    foreach(line ${lines})
        # Detect "Testing: <name>" lines that start a new test block.
        # CTest format:  "<N>/<TOTAL> Testing: <test-name>"
        if(line MATCHES "^[0-9]+/[0-9]+ Testing: (.+)$")
            set(current_test "${CMAKE_MATCH_1}")
            string(STRIP "${current_test}" current_test)
            set(capturing FALSE)
            set(output_lines "")
        endif()

        # Start collecting after the "Output:" marker.
        if(line MATCHES "^Output:$")
            set(capturing TRUE)
            continue()
        endif()

        # Collect output lines while in capturing mode.
        if(capturing)
            # The result line ends the block.
            # CTest formats:
            #   "<N>/<TOTAL> Test: <name> ........ Passed  0.12 sec"
            #   "<N>/<TOTAL> Test: <name> ....*** Failed  0.12 sec"
            if(line MATCHES "^[0-9]+/[0-9]+ Test: ")
                set(capturing FALSE)
                # Match any form of failure: "Failed", "***Failed", "FAILED"
                if(line MATCHES "[Ff]ailed|FAILED"
                   AND NOT current_test MATCHES "^${TEST_NAME}_shard[0-9]+"
                   AND NOT current_test MATCHES "_failure_summary$")
                    list(APPEND ctest_failures "${current_test}")
                    string(APPEND ctest_log
                        "==== [ctest] FAILED: ${current_test} ====\n"
                        "${output_lines}\n")
                endif()
                set(output_lines "")
            else()
                string(APPEND output_lines "${line}\n")
            endif()
        endif()
    endforeach()
endif()

# ━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━
# Write failures.log
# ━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━

set(log_path "${SHARD_DIR}/failures.log")
set(has_any_failure FALSE)

if(gtest_failures OR ctest_failures)
    set(has_any_failure TRUE)
    set(combined_log "")

    if(gtest_failures)
        string(APPEND combined_log
            "########################################\n"
            "# GTest failures (individual test cases)\n"
            "########################################\n\n"
            "${gtest_log}")
    endif()

    if(ctest_failures)
        string(APPEND combined_log
            "########################################\n"
            "# CTest failures (non-shard tests)\n"
            "########################################\n\n"
            "${ctest_log}")
    endif()

    file(WRITE "${log_path}" "${combined_log}")
else()
    file(WRITE "${log_path}" "All tests passed — no failures.\n")
endif()

# ━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━
# Console summary
# ━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━

message("\n========================================================")
message("CONSOLIDATED TEST FAILURES (all shards + ctest tests):")
message("NOTE: This summary runs before CTest prints its own exit")
message("      summary — that is normal. This is the authoritative")
message("      view of failures across all tests.")
message("========================================================")

if(gtest_failures)
    list(REMOVE_DUPLICATES gtest_failures)
    list(SORT gtest_failures)
    message("  -- GTest (per test case) --")
    foreach(f ${gtest_failures})
        message("  FAILED: ${f}")
    endforeach()
endif()

if(ctest_failures)
    list(SORT ctest_failures)
    message("  -- CTest (non-shard) --")
    foreach(f ${ctest_failures})
        message("  FAILED: ${f}")
    endforeach()
endif()

if(has_any_failure)
    message("")
    message("Full details written to artifact: build/test/gtest/failures.log")
else()
    message("  (none — all tests passed)")
endif()

message("========================================================\n")
