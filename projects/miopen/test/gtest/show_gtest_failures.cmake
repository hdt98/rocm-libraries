# show_gtest_failures.cmake
#
# Parses GTest JUnit XML files written by each shard and:
#   1. Prints a consolidated list of every failing test case to the console.
#   2. Writes a failures.log artifact containing the full error message for
#      each failing test (everything GTest printed between [ RUN ] and
#      [ FAILED ] — decoded from the XML message attribute).
#
# Called automatically by CTest via FIXTURES_CLEANUP after all shards finish,
# regardless of whether any shard passed or failed.
#
# Required variables (passed via -D on the cmake -P command line):
#   SHARD_DIR            - directory containing the per-shard XML files
#   TEST_NAME            - base name of the gtest binary (e.g. miopen_gtest)
#   GTEST_PARALLEL_LEVEL - number of shards

file(GLOB xml_files "${SHARD_DIR}/${TEST_NAME}_shard*.xml")
list(SORT xml_files)  # process shards in a stable order

set(all_failures "")
set(log_content "")

foreach(xml ${xml_files})
    if(NOT EXISTS "${xml}")
        continue()
    endif()

    # Derive a short label like "shard1" from the filename for the log header.
    get_filename_component(xml_stem "${xml}" NAME_WE)   # e.g. miopen_gtest_shard1
    string(REGEX REPLACE ".*_(shard[0-9]+)$" "\\1" shard_label "${xml_stem}")

    file(STRINGS "${xml}" lines)

    set(current_test "")
    set(in_failure FALSE)
    set(failure_msg "")

    foreach(line ${lines})
        # Capture classname + name when we enter a <testcase> element.
        if(line MATCHES "<testcase ")
            string(REGEX MATCH "classname=\"([^\"]*)\"" _ "${line}")
            set(cn "${CMAKE_MATCH_1}")
            string(REGEX MATCH " name=\"([^\"]*)\"" _ "${line}")
            set(nm "${CMAKE_MATCH_1}")
            set(current_test "${cn}.${nm}")
            set(in_failure FALSE)
            set(failure_msg "")
        endif()

        # A <failure> element means this test case failed.
        # GTest writes the full error text in the message attribute, with XML
        # entities in place of special characters (&#x0A; = newline, etc.).
        if(line MATCHES "<failure ")
            set(in_failure TRUE)

            string(REGEX MATCH "message=\"([^\"]*)\"" _ "${line}")
            set(failure_msg "${CMAKE_MATCH_1}")

            # Decode XML character entities produced by GTest's EscapeXml().
            # Order matters: &amp; must be decoded last to avoid double-decoding.
            string(REPLACE "&#x0A;" "\n"  failure_msg "${failure_msg}")
            string(REPLACE "&#x0D;" "\r"  failure_msg "${failure_msg}")
            string(REPLACE "&#x09;" "\t"  failure_msg "${failure_msg}")
            string(REPLACE "&lt;"   "<"   failure_msg "${failure_msg}")
            string(REPLACE "&gt;"   ">"   failure_msg "${failure_msg}")
            string(REPLACE "&quot;" "\""  failure_msg "${failure_msg}")
            string(REPLACE "&apos;" "'"   failure_msg "${failure_msg}")
            string(REPLACE "&amp;"  "&"   failure_msg "${failure_msg}")
        endif()

        # When we close the testcase element, flush accumulated failure info.
        if(in_failure AND line MATCHES "</testcase>")
            list(APPEND all_failures "${current_test}")

            string(APPEND log_content
                "==== [${shard_label}] FAILED: ${current_test} ====\n"
                "${failure_msg}\n"
                "\n")

            set(in_failure FALSE)
            set(failure_msg "")
        endif()
    endforeach()
endforeach()

list(REMOVE_DUPLICATES all_failures)
list(SORT all_failures)

# ── Write failures.log ──────────────────────────────────────────────────────
set(log_path "${SHARD_DIR}/failures.log")
if(all_failures)
    file(WRITE "${log_path}" "${log_content}")
else()
    file(WRITE "${log_path}" "All shards passed — no failures.\n")
endif()

# ── Console summary ──────────────────────────────────────────────────────────
message("\n========================================================")
message("CONSOLIDATED GTEST FAILURES ACROSS ALL SHARDS:")
message("========================================================")
if(all_failures)
    foreach(f ${all_failures})
        message("  FAILED: ${f}")
    endforeach()
    message("")
    message("Full error messages written to: ${log_path}")
else()
    message("  (none — all shards passed)")
endif()
message("========================================================\n")
