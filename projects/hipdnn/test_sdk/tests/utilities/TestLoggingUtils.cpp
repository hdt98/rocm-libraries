// Copyright © Advanced Micro Devices, Inc., or its affiliates.
// SPDX-License-Identifier:  MIT

#include <gtest/gtest.h>
#include <hipdnn_data_sdk/logging/LogLevel.hpp>
#include <hipdnn_data_sdk/logging/LoggingUtils.hpp>
#include <hipdnn_data_sdk/utilities/PlatformUtils.hpp>
#include <hipdnn_test_sdk/utilities/ScopedEnvironmentVariableSetter.hpp>

using namespace hipdnn_data_sdk::logging;
using namespace hipdnn_test_sdk::utilities;
using namespace hipdnn_data_sdk::utilities;

TEST(TestLoggingUtils, IsValidLogLevelWithValidLevels)
{
    // Test all valid log levels
    EXPECT_TRUE(isValidLogLevel("off"));
    EXPECT_TRUE(isValidLogLevel("info"));
    EXPECT_TRUE(isValidLogLevel("warn"));
    EXPECT_TRUE(isValidLogLevel("error"));
    EXPECT_TRUE(isValidLogLevel("fatal"));
}

TEST(TestLoggingUtils, IsValidLogLevelWithInvalidLevels)
{
    // Test invalid log levels
    EXPECT_FALSE(isValidLogLevel(""));
    EXPECT_FALSE(isValidLogLevel("debug"));
    EXPECT_FALSE(isValidLogLevel("trace"));
    EXPECT_FALSE(isValidLogLevel("verbose"));
    EXPECT_FALSE(isValidLogLevel("invalid"));
    EXPECT_FALSE(isValidLogLevel("123"));
}

TEST(TestLoggingUtils, IsValidLogLevelCaseInsensitive)
{
    // Log levels should be case-insensitive
    EXPECT_TRUE(isValidLogLevel("INFO"));
    EXPECT_TRUE(isValidLogLevel("Off"));
    EXPECT_TRUE(isValidLogLevel("ERROR"));
    EXPECT_TRUE(isValidLogLevel("Warn"));
    EXPECT_TRUE(isValidLogLevel("FATAL"));
}

TEST(TestLoggingUtils, IsValidLogLevelTrimsWhitespace)
{
    // Log levels should trim leading/trailing whitespace
    EXPECT_TRUE(isValidLogLevel(" info"));
    EXPECT_TRUE(isValidLogLevel("info "));
    EXPECT_TRUE(isValidLogLevel("  warn  "));
    EXPECT_TRUE(isValidLogLevel("\terror\t"));
    EXPECT_TRUE(isValidLogLevel("\n off \n"));
}

TEST(TestLoggingUtils, IsLoggingEnabledWithValidLevels)
{
    ScopedEnvironmentVariableSetter guard("HIPDNN_LOG_LEVEL");

    guard.setValue("off");
    EXPECT_FALSE(isLoggingEnabled());

    guard.setValue("info");
    EXPECT_TRUE(isLoggingEnabled());

    guard.setValue("error");
    EXPECT_TRUE(isLoggingEnabled());
}

TEST(TestLoggingUtils, IsLoggingEnabledWithInvalidOrUnsetLevels)
{
    ScopedEnvironmentVariableSetter guard("HIPDNN_LOG_LEVEL", "invalid");

    EXPECT_FALSE(isLoggingEnabled());

    hipdnn_data_sdk::utilities::unsetEnv("HIPDNN_LOG_LEVEL");
    EXPECT_FALSE(isLoggingEnabled());
}

// ============================================================================
// Tests for stringToSeverity() - returns std::optional<hipdnnSeverity_t>
// ============================================================================

TEST(TestStringToSeverity, ValidOffReturnsOptionalWithOff)
{
    auto result = detail::stringToSeverity("off");
    ASSERT_TRUE(result.has_value());
    EXPECT_EQ(
        *result,
        HIPDNN_SEV_OFF); // NOLINT(bugprone-unchecked-optional-access) - checked by ASSERT_TRUE
}

TEST(TestStringToSeverity, ValidInfoReturnsOptionalWithInfo)
{
    auto result = detail::stringToSeverity("info");
    ASSERT_TRUE(result.has_value());
    EXPECT_EQ(
        *result,
        HIPDNN_SEV_INFO); // NOLINT(bugprone-unchecked-optional-access) - checked by ASSERT_TRUE
}

TEST(TestStringToSeverity, ValidWarnReturnsOptionalWithWarn)
{
    auto result = detail::stringToSeverity("warn");
    ASSERT_TRUE(result.has_value());
    EXPECT_EQ(
        *result,
        HIPDNN_SEV_WARN); // NOLINT(bugprone-unchecked-optional-access) - checked by ASSERT_TRUE
}

TEST(TestStringToSeverity, ValidErrorReturnsOptionalWithError)
{
    auto result = detail::stringToSeverity("error");
    ASSERT_TRUE(result.has_value());
    EXPECT_EQ(
        *result,
        HIPDNN_SEV_ERROR); // NOLINT(bugprone-unchecked-optional-access) - checked by ASSERT_TRUE
}

TEST(TestStringToSeverity, ValidFatalReturnsOptionalWithFatal)
{
    auto result = detail::stringToSeverity("fatal");
    ASSERT_TRUE(result.has_value());
    EXPECT_EQ(
        *result,
        HIPDNN_SEV_FATAL); // NOLINT(bugprone-unchecked-optional-access) - checked by ASSERT_TRUE
}

TEST(TestStringToSeverity, InvalidStringReturnsNullopt)
{
    EXPECT_FALSE(detail::stringToSeverity("invalid").has_value());
    EXPECT_FALSE(detail::stringToSeverity("debug").has_value());
    EXPECT_FALSE(detail::stringToSeverity("trace").has_value());
    EXPECT_FALSE(detail::stringToSeverity("verbose").has_value());
    EXPECT_FALSE(detail::stringToSeverity("123").has_value());
}

TEST(TestStringToSeverity, EmptyStringReturnsNullopt)
{
    EXPECT_FALSE(detail::stringToSeverity("").has_value());
}

TEST(TestStringToSeverity, CaseInsensitiveMatching)
{
    // Uppercase
    ASSERT_TRUE(detail::stringToSeverity("OFF").has_value());
    EXPECT_EQ(
        *detail::stringToSeverity("OFF"),
        HIPDNN_SEV_OFF); // NOLINT(bugprone-unchecked-optional-access) - checked by ASSERT_TRUE

    ASSERT_TRUE(detail::stringToSeverity("INFO").has_value());
    EXPECT_EQ(
        *detail::stringToSeverity("INFO"),
        HIPDNN_SEV_INFO); // NOLINT(bugprone-unchecked-optional-access) - checked by ASSERT_TRUE

    ASSERT_TRUE(detail::stringToSeverity("WARN").has_value());
    EXPECT_EQ(
        *detail::stringToSeverity("WARN"),
        HIPDNN_SEV_WARN); // NOLINT(bugprone-unchecked-optional-access) - checked by ASSERT_TRUE

    ASSERT_TRUE(detail::stringToSeverity("ERROR").has_value());
    EXPECT_EQ(
        *detail::stringToSeverity("ERROR"),
        HIPDNN_SEV_ERROR); // NOLINT(bugprone-unchecked-optional-access) - checked by ASSERT_TRUE

    ASSERT_TRUE(detail::stringToSeverity("FATAL").has_value());
    EXPECT_EQ(
        *detail::stringToSeverity("FATAL"),
        HIPDNN_SEV_FATAL); // NOLINT(bugprone-unchecked-optional-access) - checked by ASSERT_TRUE

    // Mixed case
    ASSERT_TRUE(detail::stringToSeverity("Info").has_value());
    EXPECT_EQ(
        *detail::stringToSeverity("Info"),
        HIPDNN_SEV_INFO); // NOLINT(bugprone-unchecked-optional-access) - checked by ASSERT_TRUE

    ASSERT_TRUE(detail::stringToSeverity("WaRn").has_value());
    EXPECT_EQ(
        *detail::stringToSeverity("WaRn"),
        HIPDNN_SEV_WARN); // NOLINT(bugprone-unchecked-optional-access) - checked by ASSERT_TRUE
}

TEST(TestStringToSeverity, TrimsWhitespace)
{
    // Leading whitespace
    ASSERT_TRUE(detail::stringToSeverity("  info").has_value());
    EXPECT_EQ(
        *detail::stringToSeverity("  info"),
        HIPDNN_SEV_INFO); // NOLINT(bugprone-unchecked-optional-access) - checked by ASSERT_TRUE

    // Trailing whitespace
    ASSERT_TRUE(detail::stringToSeverity("warn  ").has_value());
    EXPECT_EQ(
        *detail::stringToSeverity("warn  "),
        HIPDNN_SEV_WARN); // NOLINT(bugprone-unchecked-optional-access) - checked by ASSERT_TRUE

    // Both ends
    ASSERT_TRUE(detail::stringToSeverity("  error  ").has_value());
    EXPECT_EQ(
        *detail::stringToSeverity("  error  "),
        HIPDNN_SEV_ERROR); // NOLINT(bugprone-unchecked-optional-access) - checked by ASSERT_TRUE

    // Tabs and newlines
    ASSERT_TRUE(detail::stringToSeverity("\tfatal\n").has_value());
    EXPECT_EQ(
        *detail::stringToSeverity("\tfatal\n"),
        HIPDNN_SEV_FATAL); // NOLINT(bugprone-unchecked-optional-access) - checked by ASSERT_TRUE

    // Whitespace-only still invalid
    EXPECT_FALSE(detail::stringToSeverity("   ").has_value());
}

// ============================================================================
// Tests for stringToSeverityOrOff() - returns hipdnnSeverity_t (OFF on invalid)
// ============================================================================

TEST(TestStringToSeverityOrOff, ValidInputsReturnCorrectEnum)
{
    EXPECT_EQ(detail::stringToSeverityOrOff("off"), HIPDNN_SEV_OFF);
    EXPECT_EQ(detail::stringToSeverityOrOff("info"), HIPDNN_SEV_INFO);
    EXPECT_EQ(detail::stringToSeverityOrOff("warn"), HIPDNN_SEV_WARN);
    EXPECT_EQ(detail::stringToSeverityOrOff("error"), HIPDNN_SEV_ERROR);
    EXPECT_EQ(detail::stringToSeverityOrOff("fatal"), HIPDNN_SEV_FATAL);
}

TEST(TestStringToSeverityOrOff, InvalidInputReturnsOff)
{
    EXPECT_EQ(detail::stringToSeverityOrOff("invalid"), HIPDNN_SEV_OFF);
    EXPECT_EQ(detail::stringToSeverityOrOff("debug"), HIPDNN_SEV_OFF);
    EXPECT_EQ(detail::stringToSeverityOrOff("trace"), HIPDNN_SEV_OFF);
    EXPECT_EQ(detail::stringToSeverityOrOff("123"), HIPDNN_SEV_OFF);
}

TEST(TestStringToSeverityOrOff, EmptyStringReturnsOff)
{
    EXPECT_EQ(detail::stringToSeverityOrOff(""), HIPDNN_SEV_OFF);
}

TEST(TestStringToSeverityOrOff, CaseInsensitiveAndTrimsWhitespace)
{
    EXPECT_EQ(detail::stringToSeverityOrOff("  INFO  "), HIPDNN_SEV_INFO);
    EXPECT_EQ(detail::stringToSeverityOrOff("\tWARN\n"), HIPDNN_SEV_WARN);
    EXPECT_EQ(detail::stringToSeverityOrOff("Error"), HIPDNN_SEV_ERROR);
}
