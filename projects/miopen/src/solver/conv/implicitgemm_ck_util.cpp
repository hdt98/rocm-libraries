/*******************************************************************************
 *
 * MIT License
 *
 * Copyright (c) 2023 Advanced Micro Devices, Inc.
 *
 * Permission is hereby granted, free of charge, to any person obtaining a copy
 * of this software and associated documentation files (the "Software"), to deal
 * in the Software without restriction, including without limitation the rights
 * to use, copy, modify, merge, publish, distribute, sublicense, and/or sell
 * copies of the Software, and to permit persons to whom the Software is
 * furnished to do so, subject to the following conditions:
 *
 * The above copyright notice and this permission notice shall be included in all
 * copies or substantial portions of the Software.
 *
 * THE SOFTWARE IS PROVIDED "AS IS", WITHOUT WARRANTY OF ANY KIND, EXPRESS OR
 * IMPLIED, INCLUDING BUT NOT LIMITED TO THE WARRANTIES OF MERCHANTABILITY,
 * FITNESS FOR A PARTICULAR PURPOSE AND NONINFRINGEMENT. IN NO EVENT SHALL THE
 * AUTHORS OR COPYRIGHT HOLDERS BE LIABLE FOR ANY CLAIM, DAMAGES OR OTHER
 * LIABILITY, WHETHER IN AN ACTION OF CONTRACT, TORT OR OTHERWISE, ARISING FROM,
 * OUT OF OR IN CONNECTION WITH THE SOFTWARE OR THE USE OR OTHER DEALINGS IN THE
 * SOFTWARE.
 *
 *******************************************************************************/

//#include <miopen/solver/implicitgemm_ck_util.hpp>

#include <origami/types.hpp>
#include <vector>

namespace miopen {
namespace solver {

// Helper function to trim whitespace from both ends of a string
std::string trim(const std::string& str)
{
    size_t first = str.find_first_not_of(" \t\n\r");
    if(first == std::string::npos)
        return "";
    size_t last = str.find_last_not_of(" \t\n\r");
    return str.substr(first, (last - first + 1));
}

/**
 * Parse template arguments from a device instance definition string.
 *
 * This function extracts and parses the template arguments from a device instance
 * definition, properly handling nested angle brackets (e.g., S<4, 8, 1>).
 *
 * @param instance_str A string containing a device instance definition
 *
 * Example input formats:
 *   "DeviceGroupedConvBwdDataMultipleD_Xdl_CShuffle_v1< NDimSpatial, ALayout, BLayout, S<4, 8, 1>,
 * 64, 16, ...>" "DeviceGroupedGemmXdlSplitKCShuffle<Row, Row, Row, Row, F16, F16, F32, F16,
 * PassThrough, PassThrough, ...>"
 *
 * @return Vector of strings, each containing one template argument (trimmed of whitespace)
 *
 * Example usage:
 *   std::string inst = "DeviceKernel< int, float, S<1, 2, 3>, true>";
 *   auto args = parseDeviceInstanceArguments(inst);
 *   // args[0] = "DeviceKernel"
 *   // args[1] = "int"
 *   // args[2] = "float"
 *   // args[3] = "S<1, 2, 3>"
 *   // args[4] = "true"
 */
std::vector<std::string> parseDeviceInstanceArguments(const std::string& instance_str)
{
    std::vector<std::string> arguments;

    // Find the first '<' which starts the template arguments
    size_t start_pos = instance_str.find('<');
    if(start_pos == std::string::npos)
    {
        return arguments; // No template arguments found
    }

    std::string kernel_name = trim(instance_str.substr(0, start_pos));
    arguments.push_back(kernel_name);

    // Find the matching '>' for the outermost template
    int bracket_depth = 0;
    size_t end_pos    = start_pos;
    for(size_t i = start_pos; i < instance_str.length(); ++i)
    {
        if(instance_str[i] == '<')
        {
            bracket_depth++;
        }
        else if(instance_str[i] == '>')
        {
            bracket_depth--;
            if(bracket_depth == 0)
            {
                end_pos = i;
                break;
            }
        }
    }

    if(bracket_depth != 0 || end_pos == start_pos)
    {
        return arguments; // Malformed template arguments
    }

    // Extract the template arguments substring (between < and >)
    std::string args_str = instance_str.substr(start_pos + 1, end_pos - start_pos - 1);

    // Parse arguments, respecting nested angle brackets
    bracket_depth    = 0;
    size_t arg_start = 0;

    for(size_t i = 0; i < args_str.length(); ++i)
    {
        char c = args_str[i];

        if(c == '<')
        {
            bracket_depth++;
        }
        else if(c == '>')
        {
            bracket_depth--;
        }
        else if(c == ',' && bracket_depth == 0)
        {
            // Found a comma at the top level - this separates arguments
            std::string arg = trim(args_str.substr(arg_start, i - arg_start));
            if(!arg.empty())
            {
                arguments.push_back(arg);
            }
            arg_start = i + 1;
        }
    }

    // Don't forget the last argument
    std::string last_arg = trim(args_str.substr(arg_start));
    if(!last_arg.empty())
    {
        arguments.push_back(last_arg);
    }

    return arguments;
}

int IndexOf(const std::vector<std::string>& vec, const std::string& val)
{
    auto it = std::find(vec.begin(), vec.end(), val);
    if(it != vec.end())
    {
        return std::distance(vec.begin(), it);
    }
    return -1;
}

std::vector<std::string> parseXSplit(const std::string& line, const size_t len)
{
    std::string parsing = line;
    std::vector<std::string> vals{};
    size_t val_end = 0, line_end = len;
    while(line_end > 0)
    {
        val_end = parsing.find('x');
        if(val_end > line_end || val_end == std::string::npos)
            val_end = line_end;
        line_end -= std::min(line_end, (val_end + 1));

        vals.push_back(std::string(parsing.substr(0, val_end)));
        if(line_end > 0)
            parsing = parsing.substr(val_end + 1);
    }
    return vals;
}

std::vector<std::string> parseDeviceInstanceInlineArgs(const std::string& instance_str)
{
    // DeviceGroupedConvBwdWeight_Explicit_Xdl<DeviceBatchedGemmXdlUniversal<MNKPadding, CRR>
    // BlkSize: 256, BlkTile: 256x16x64, WaveTile: 16x16, WaveMap: 4x1, VmemReadVec: 2x2,
    // BlkGemmPipelineScheduler: Intrawave, BlkGemmPipelineVersion: v2,
    // BlkGemmPipelinePrefetchStages: 2> fields = {"MPerBlock", "NPerBlock", "KPerBlock", "MPerOp",
    // "NPerOp"};

    size_t start_pos, len;
    std::string parsing;
    start_pos       = instance_str.find("BlkTile: ");
    parsing         = instance_str.substr(start_pos + 9);
    len             = parsing.find(',');
    auto block_tile = parseXSplit(parsing, len);

    start_pos      = instance_str.find("WaveTile: ");
    parsing        = instance_str.substr(start_pos + 10);
    len            = parsing.find(',');
    auto wave_tile = parseXSplit(parsing, len);

    auto ret = block_tile;
    for(auto val : wave_tile)
        ret.push_back(val);

    return ret;
}

} // namespace solver
} // namespace miopen
