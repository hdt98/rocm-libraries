/*******************************************************************************
 *
 * MIT License
 *
 * Copyright 2024-2025 AMD ROCm(TM) Software
 *
 * Permission is hereby granted, free of charge, to any person obtaining a copy
 * of this software and associated documentation files (the "Software"), to deal
 * in the Software without restriction, including without limitation the rights
 * to use, copy, modify, merge, publish, distribute, sublicense, and/or sell
 * copies of the Software, and to permit persons to whom the Software is
 * furnished to do so, subject to the following conditions:
 *
 * The above copyright notice and this permission notice shall be included in
 * all copies or substantial portions of the Software.
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
// ...existing code...
#include <rocRoller/Utilities/Error.hpp>
#include <fstream>
#include <sstream>
#include <vector>

#include <amd_comgr.h>

using namespace rocRoller;

std::string rocRoller::readMetaDataFromCodeObject(std::string const& fileName)
{
    std::string yaml;
    // Read the file into memory
    std::ifstream file(fileName, std::ios::binary | std::ios::ate);
    if(!file.is_open())
    {
        Throw<FatalError>("Error opening file: ", fileName);
    }
    
    std::streamsize size = file.tellg();
    file.seekg(0, std::ios::beg);
    
    std::vector<char> buffer(size);
    if(!file.read(buffer.data(), size))
    {
        Throw<FatalError>("Error reading file: ", fileName);
    }
    file.close();
    
    // Create comgr data object
    amd_comgr_data_t dataIn;
    amd_comgr_status_t status = amd_comgr_create_data(AMD_COMGR_DATA_KIND_RELOCATABLE, &dataIn);
    if(status != AMD_COMGR_STATUS_SUCCESS)
    {
        const char* statusString;
        amd_comgr_status_string(status, &statusString);
        Throw<FatalError>("Failed to create data object: ", statusString);
    }
    
    // Set the data
    status = amd_comgr_set_data(dataIn, size, buffer.data());
    if(status != AMD_COMGR_STATUS_SUCCESS)
    {
        const char* statusString;
        amd_comgr_status_string(status, &statusString);
        amd_comgr_release_data(dataIn);
        Throw<FatalError>("Failed to set data: ", statusString);
    }
    
    // Get metadata from data object
    amd_comgr_metadata_node_t meta;
    status = amd_comgr_get_data_metadata(dataIn, &meta);
    if(status != AMD_COMGR_STATUS_SUCCESS)
    {
        const char* statusString;
        amd_comgr_status_string(status, &statusString);
        amd_comgr_release_data(dataIn);
        Throw<FatalError>("Failed to get metadata: ", statusString);
    }
    
    // Check if we have valid metadata
    amd_comgr_metadata_kind_t mkind;
    status = amd_comgr_get_metadata_kind(meta, &mkind);
    if(status == AMD_COMGR_STATUS_SUCCESS && mkind != AMD_COMGR_METADATA_KIND_NULL)
    {
        // Convert metadata to YAML by iterating through the map
        // Since comgr doesn't provide a direct toYAML function, we'll need to
        // manually build the YAML string by traversing the metadata structure
        std::ostringstream yamlStream;
        
        // Helper lambda to convert metadata to YAML format
        std::function<void(amd_comgr_metadata_node_t, int)> metadataToYaml;
        metadataToYaml = [&](amd_comgr_metadata_node_t node, int indent) {
            amd_comgr_metadata_kind_t kind;
            if(amd_comgr_get_metadata_kind(node, &kind) != AMD_COMGR_STATUS_SUCCESS)
                return;
                
            std::string indentStr(indent * 2, ' ');
            
            switch(kind)
            {
                case AMD_COMGR_METADATA_KIND_STRING:
                {
                    size_t size = 0;
                    amd_comgr_get_metadata_string(node, &size, nullptr);
                    std::vector<char> str(size);
                    amd_comgr_get_metadata_string(node, &size, str.data());
                    yamlStream << str.data();
                    break;
                }
                case AMD_COMGR_METADATA_KIND_MAP:
                {
                    struct MapContext {
                        std::ostringstream* stream;
                        std::function<void(amd_comgr_metadata_node_t, int)>* converter;
                        int indent;
                        bool first;
                    };
                    
                    MapContext ctx = { &yamlStream, &metadataToYaml, indent, true };
                    
                    amd_comgr_iterate_map_metadata(node, 
                        [](amd_comgr_metadata_node_t key, amd_comgr_metadata_node_t value, void* user_data) -> amd_comgr_status_t {
                            MapContext* ctx = static_cast<MapContext*>(user_data);
                            
                            if(!ctx->first)
                                *(ctx->stream) << "\n";
                            ctx->first = false;
                            
                            std::string indentStr(ctx->indent * 2, ' ');
                            *(ctx->stream) << indentStr;
                            
                            // Output key
                            size_t keySize = 0;
                            amd_comgr_get_metadata_string(key, &keySize, nullptr);
                            std::vector<char> keyStr(keySize);
                            amd_comgr_get_metadata_string(key, &keySize, keyStr.data());
                            *(ctx->stream) << keyStr.data() << ": ";
                            
                            // Check value kind
                            amd_comgr_metadata_kind_t valueKind;
                            amd_comgr_get_metadata_kind(value, &valueKind);
                            
                            if(valueKind == AMD_COMGR_METADATA_KIND_MAP || valueKind == AMD_COMGR_METADATA_KIND_LIST)
                            {
                                *(ctx->stream) << "\n";
                                (*ctx->converter)(value, ctx->indent + 1);
                            }
                            else
                            {
                                (*ctx->converter)(value, 0);
                            }
                            
                            return AMD_COMGR_STATUS_SUCCESS;
                        }, &ctx);
                    break;
                }
                case AMD_COMGR_METADATA_KIND_LIST:
                {
                    size_t listSize = 0;
                    amd_comgr_get_metadata_list_size(node, &listSize);
                    
                    for(size_t i = 0; i < listSize; ++i)
                    {
                        yamlStream << indentStr << "- ";
                        
                        amd_comgr_metadata_node_t item;
                        if(amd_comgr_index_list_metadata(node, i, &item) == AMD_COMGR_STATUS_SUCCESS)
                        {
                            amd_comgr_metadata_kind_t itemKind;
                            amd_comgr_get_metadata_kind(item, &itemKind);
                            
                            if(itemKind == AMD_COMGR_METADATA_KIND_MAP || itemKind == AMD_COMGR_METADATA_KIND_LIST)
                            {
                                yamlStream << "\n";
                                metadataToYaml(item, indent + 1);
                            }
                            else
                            {
                                metadataToYaml(item, 0);
                            }
                            
                            amd_comgr_destroy_metadata(item);
                        }
                        
                        if(i < listSize - 1)
                            yamlStream << "\n";
                    }
                    break;
                }
                default:
                    break;
            }
        };
        
        if(mkind == AMD_COMGR_METADATA_KIND_MAP)
        {
            metadataToYaml(meta, 0);
            yaml = yamlStream.str();
        }
    }
    
    // Cleanup
    amd_comgr_destroy_metadata(meta);
    amd_comgr_release_data(dataIn);

    return yaml;
}