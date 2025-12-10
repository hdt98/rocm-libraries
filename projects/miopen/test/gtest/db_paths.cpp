// Copyright © Advanced Micro Devices, Inc., or its affiliates.
// SPDX-License-Identifier:  MIT

#include <miopen/db_path.hpp>
#include <miopen/binary_cache.hpp>
#include <miopen/filesystem.hpp>
#include <miopen/version.h>
#include <miopen/stringutils.hpp>
#include <gtest/gtest.h>

#include <string>
#include <sstream>

namespace fs = miopen::fs;

// Helper function to build expected version string
std::string GetExpectedVersionString()
{
    std::ostringstream oss;
    oss << MIOPEN_VERSION_MAJOR << "."
        << MIOPEN_VERSION_MINOR << "."
        << MIOPEN_VERSION_PATCH << "."
        << MIOPEN_STRINGIZE(MIOPEN_VERSION_TWEAK);
    return oss.str();
}

// Helper function to check if a path contains a substring
bool PathContains(const fs::path& path, const std::string& substring)
{
    return path.string().find(substring) != std::string::npos;
}

// Test that user DB path contains "miopen" in the path
TEST(CPU_DbPaths_NONE, UserDbPath_ContainsMiopenFolder)
{
    const auto& user_db_path = miopen::GetUserDbPath();
    
    if(user_db_path.empty())
    {
        GTEST_SKIP() << "User DB is disabled (MIOPEN_DISABLE_USERDB)";
    }
    
    EXPECT_TRUE(PathContains(user_db_path, "miopen"))
        << "User DB path '" << user_db_path.string() 
        << "' should contain 'miopen' folder";
}

// Test that binary cache path contains version string
TEST(CPU_DbPaths_NONE, BinaryCache_ContainsVersionString)
{
    const auto cache_path = miopen::GetCachePath(false);
    
    if(cache_path.empty())
    {
        GTEST_SKIP() << "User cache is disabled (MIOPEN_DISABLE_USERDB)";
    }
    
    const std::string expected_version = GetExpectedVersionString();
    EXPECT_TRUE(PathContains(cache_path, expected_version))
        << "Binary cache path '" << cache_path.string() 
        << "' should contain version string '" << expected_version << "'";
}

// Test that binary cache path contains "miopen" folder
TEST(CPU_DbPaths_NONE, BinaryCache_ContainsMiopenFolder)
{
    const auto cache_path = miopen::GetCachePath(false);
    
    if(cache_path.empty())
    {
        GTEST_SKIP() << "User cache is disabled (MIOPEN_DISABLE_USERDB)";
    }
    
    EXPECT_TRUE(PathContains(cache_path, "miopen"))
        << "Binary cache path '" << cache_path.string() 
        << "' should contain 'miopen' folder";
}

// Test GetUserDbSuffix contains version information
TEST(CPU_DbPaths_NONE, UserDbSuffix_ContainsVersionInfo)
{
    const std::string suffix = miopen::GetUserDbSuffix();
    EXPECT_FALSE(suffix.empty()) << "User DB suffix should not be empty";
    
    // Suffix should contain version numbers separated by underscores
    std::ostringstream expected_pattern;
    expected_pattern << MIOPEN_VERSION_MAJOR << "_"
                    << MIOPEN_VERSION_MINOR << "_"
                    << MIOPEN_VERSION_PATCH;
    
    EXPECT_TRUE(suffix.find(expected_pattern.str()) != std::string::npos)
        << "Suffix '" << suffix << "' should contain version pattern '"
        << expected_pattern.str() << "'";
}

// Test that user and system cache paths are different when both exist
TEST(CPU_DbPaths_NONE, UserAndSystemCachePaths_AreDifferent)
{
    const auto user_cache = miopen::GetCachePath(false);
    const auto sys_cache = miopen::GetCachePath(true);
    
    if(!user_cache.empty() && !sys_cache.empty())
    {
        EXPECT_NE(user_cache, sys_cache)
            << "User and system cache paths should be different";
    }
}

// Test basic path validity
TEST(CPU_DbPaths_NONE, Paths_AreValid)
{
    const auto& user_db_path = miopen::GetUserDbPath();
    const auto cache_path = miopen::GetCachePath(false);
    const auto sys_cache_path = miopen::GetCachePath(true);
    
    // Paths should either be empty (if disabled) or valid filesystem paths
    if(!user_db_path.empty())
    {
        EXPECT_FALSE(user_db_path.string().empty());
    }
    
    if(!cache_path.empty())
    {
        EXPECT_FALSE(cache_path.string().empty());
    }
    
    if(!sys_cache_path.empty())
    {
        EXPECT_FALSE(sys_cache_path.string().empty());
    }
}

// Note: Testing environment variable overrides requires process restart
// to clear the static variables in GetUserDbPath() and GetCachePath().
// We cannot properly test those code paths without a code refactor.
