# show_gtest_failures.cmake
#
# Parses two sources of test result XML and writes failures.log:
#
#   1. Per-shard GTest JUnit XML (miopen_gtest_shard*.xml)
#      Written by --gtest_output=xml.  The <failure message="..."> attribute
#      holds the full assertion text (file:line + values), encoded with XML
#      character entities.  This gives individual GTest-case granularity.
#
#   2. CTest-level JUnit XML (ctest_junit_results.xml)
#      Written by ctest --output-junit.  The <system-out> element holds the
#      captured stdout of each CTest test.  Shard entries are skipped here
#      because they are already covered by source 1 above.
#
# Called automatically by CTest via FIXTURES_CLEANUP after all shards finish,
# regardless of whether any shard passed or failed.
#
# Required variables (passed via -D on the cmake -P command line):
#   SHARD_DIR            - directory containing the per-shard GTest XML files
#   TEST_NAME            - base name of the gtest binary (e.g. miopen_gtest)
#   GTEST_PARALLEL_LEVEL - number of shards
#
# The CTest JUnit XML is expected one level above SHARD_DIR (i.e. the project
# binary directory), which is where CMAKE_CTEST_ARGUMENTS points it.

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
# Part 2 — CTest-level JUnit XML (non-shard CTest tests)
# ━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━

# The CTest JUnit XML is written to the project binary dir (one level up from
# the gtest build dir where SHARD_DIR points).
get_filename_component(project_binary_dir "${SHARD_DIR}" DIRECTORY)
get_filename_component(project_binary_dir "${project_binary_dir}" DIRECTORY)
set(ctest_junit "${project_binary_dir}/ctest_junit_results.xml")

set(ctest_failures "")   # list of test names
set(ctest_log "")        # log content for ctest section

if(EXISTS "${ctest_junit}")
    file(STRINGS "${ctest_junit}" lines)

    set(current_test "")
    set(is_failed FALSE)
    set(in_sysout FALSE)
    set(sysout_content "")

    foreach(line ${lines})
        # New testcase element — reset state.
        if(line MATCHES "<testcase ")
            string(REGEX MATCH " name=\"([^\"]*)\"" _ "${line}")
            set(current_test "${CMAKE_MATCH_1}")
            set(is_failed FALSE)
            set(in_sysout FALSE)
            set(sysout_content "")
        endif()

        # A <failure> element marks the test as failed.
        # (The message attribute only says "Failed" for CTest tests; the real
        # output is in <system-out> below.)
        if(line MATCHES "<failure")
            set(is_failed TRUE)
        endif()

        # Collect lines inside <system-out>...</system-out>.
        if(line MATCHES "<system-out>")
            set(in_sysout TRUE)
            # Grab any content on the same line after the tag.
            string(REGEX REPLACE ".*<system-out>" "" rest "${line}")
            if(NOT rest MATCHES "^[[:space:]]*$")
                string(APPEND sysout_content "${rest}\n")
            endif()
        elseif(in_sysout)
            if(line MATCHES "</system-out>")
                set(in_sysout FALSE)
            else()
                string(APPEND sysout_content "${line}\n")
            endif()
        endif()

        # End of testcase — record if it failed and is not a shard
        # (shards are already covered by Part 1).
        if(line MATCHES "</testcase>" AND is_failed)
            if(NOT current_test MATCHES "^${TEST_NAME}_shard[0-9]+$"
               AND NOT current_test MATCHES "_failure_summary$")
                list(APPEND ctest_failures "${current_test}")
                string(APPEND ctest_log
                    "==== [ctest] FAILED: ${current_test} ====\n"
                    "${sysout_content}\n")
            endif()
            set(is_failed FALSE)
            set(sysout_content "")
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
message("CONSOLIDATED TEST FAILURES:")
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
    message("Full output written to: ${log_path}")
else()
    message("  (none — all tests passed)")
endif()

message("========================================================\n")
