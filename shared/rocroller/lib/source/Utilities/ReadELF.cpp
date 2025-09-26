#include <rocRoller/Serialization/comgr/comgr.hpp>
#include <rocRoller/Utilities/Error.hpp>

#include <amd_comgr/amd_comgr.h>
#include <yaml-cpp/yaml.h>

#include <fstream>
#include <memory>
#include <sstream>

using namespace rocRoller;
using namespace rocRoller::Serialization;

namespace
{
    // RAII wrapper for COMGR data objects
    class COMGRDataHandle
    {
    private:
        amd_comgr_data_t data{0};
        bool             owned = false;

    public:
        COMGRDataHandle() = default;

        ~COMGRDataHandle()
        {
            if(owned && data.handle != 0)
            {
                amd_comgr_release_data(data);
            }
        }

        // Move constructor
        COMGRDataHandle(COMGRDataHandle&& other) noexcept
            : data(other.data)
            , owned(other.owned)
        {
            other.data.handle = 0;
            other.owned       = false;
        }

        // Move assignment
        COMGRDataHandle& operator=(COMGRDataHandle&& other) noexcept
        {
            if(this != &other)
            {
                if(owned && data.handle != 0)
                {
                    amd_comgr_release_data(data);
                }
                data              = other.data;
                owned             = other.owned;
                other.data.handle = 0;
                other.owned       = false;
            }
            return *this;
        }

        // Delete copy constructor and assignment
        COMGRDataHandle(const COMGRDataHandle&) = delete;
        COMGRDataHandle& operator=(const COMGRDataHandle&) = delete;

        amd_comgr_data_t get() const
        {
            return data;
        }

        amd_comgr_data_t* getPtr()
        {
            return &data;
        }

        void setOwned(bool own)
        {
            owned = own;
        }

        bool isValid() const
        {
            return data.handle != 0;
        }
    };

    // Convert COMGR metadata to YAML recursively
    void metadataToYAML(amd_comgr_metadata_node_t node, YAML::Emitter& out)
    {
        auto kind = getMetadataKind(node);

        switch(kind)
        {
        case AMD_COMGR_METADATA_KIND_NULL:
            out << YAML::Null;
            break;

        case AMD_COMGR_METADATA_KIND_STRING:
        {
            auto str = getMetadataString(node);
            out << str;
            break;
        }

        case AMD_COMGR_METADATA_KIND_MAP:
        {
            out << YAML::BeginMap;

            struct MapContext
            {
                YAML::Emitter* emitter;
            };

            MapContext ctx{&out};

            auto callback = [](amd_comgr_metadata_node_t key,
                               amd_comgr_metadata_node_t value,
                               void*                     user_data) -> amd_comgr_status_t {
                auto* ctx     = static_cast<MapContext*>(user_data);
                auto  key_str = getMetadataString(key);

                *(ctx->emitter) << YAML::Key << key_str << YAML::Value;
                metadataToYAML(value, *(ctx->emitter));

                return AMD_COMGR_STATUS_SUCCESS;
            };

            amd_comgr_iterate_map_metadata(node, callback, &ctx);

            out << YAML::EndMap;
            break;
        }

        case AMD_COMGR_METADATA_KIND_LIST:
        {
            out << YAML::BeginSeq;

            size_t count = 0;
            amd_comgr_get_metadata_list_size(node, &count);

            for(size_t i = 0; i < count; i++)
            {
                amd_comgr_metadata_node_t elem   = {0};
                auto                      status = amd_comgr_index_list_metadata(node, i, &elem);
                if(status == AMD_COMGR_STATUS_SUCCESS)
                {
                    MetadataNodeHandle elem_handle(elem);
                    metadataToYAML(elem_handle.get(), out);
                }
            }

            out << YAML::EndSeq;
            break;
        }
        }
    }

    // Read file contents into vector
    std::vector<char> readFileToVector(const std::string& fileName)
    {
        std::ifstream file(fileName, std::ios::binary | std::ios::ate);
        if(!file.is_open())
        {
            Throw<FatalError>("Failed to open file: ", fileName);
        }

        auto              size = file.tellg();
        std::vector<char> buffer(size);

        file.seekg(0, std::ios::beg);
        file.read(buffer.data(), size);

        if(!file)
        {
            Throw<FatalError>("Failed to read file: ", fileName);
        }

        return buffer;
    }
}

std::string rocRoller::readMetaDataFromCodeObject(std::string const& fileName)
{
    std::string yaml;

    // Read file contents
    auto fileData = readFileToVector(fileName);

    // Create COMGR data object for the code object
    COMGRDataHandle codeObjectData;
    auto status = amd_comgr_create_data(AMD_COMGR_DATA_KIND_EXECUTABLE, codeObjectData.getPtr());
    if(status != AMD_COMGR_STATUS_SUCCESS)
    {
        Throw<FatalError>("Failed to create COMGR data object: ", status);
    }
    codeObjectData.setOwned(true);

    // Set the data from file contents
    status = amd_comgr_set_data(codeObjectData.get(), fileData.size(), fileData.data());
    if(status != AMD_COMGR_STATUS_SUCCESS)
    {
        Throw<FatalError>("Failed to set COMGR data: ", status);
    }

    // Get metadata from the code object
    amd_comgr_metadata_node_t metadata = {0};
    status = amd_comgr_get_data_metadata(codeObjectData.get(), &metadata);
    if(status != AMD_COMGR_STATUS_SUCCESS)
    {
        // Try other data kinds if executable fails
        codeObjectData = COMGRDataHandle();

        // Try as relocatable
        status = amd_comgr_create_data(AMD_COMGR_DATA_KIND_RELOCATABLE, codeObjectData.getPtr());
        if(status == AMD_COMGR_STATUS_SUCCESS)
        {
            codeObjectData.setOwned(true);
            status = amd_comgr_set_data(codeObjectData.get(), fileData.size(), fileData.data());
            if(status == AMD_COMGR_STATUS_SUCCESS)
            {
                status = amd_comgr_get_data_metadata(codeObjectData.get(), &metadata);
            }
        }

        if(status != AMD_COMGR_STATUS_SUCCESS)
        {
            // Try as BC (bitcode)
            codeObjectData = COMGRDataHandle();
            status         = amd_comgr_create_data(AMD_COMGR_DATA_KIND_BC, codeObjectData.getPtr());
            if(status == AMD_COMGR_STATUS_SUCCESS)
            {
                codeObjectData.setOwned(true);
                status = amd_comgr_set_data(codeObjectData.get(), fileData.size(), fileData.data());
                if(status == AMD_COMGR_STATUS_SUCCESS)
                {
                    status = amd_comgr_get_data_metadata(codeObjectData.get(), &metadata);
                }
            }
        }

        if(status != AMD_COMGR_STATUS_SUCCESS)
        {
            // If still no success, return empty YAML
            return yaml;
        }
    }

    // Ensure we clean up the metadata node
    MetadataNodeHandle metadataHandle(metadata);

    // Check if metadata is valid
    auto kind = getMetadataKind(metadata);
    if(kind == AMD_COMGR_METADATA_KIND_NULL)
    {
        // No metadata found
        return yaml;
    }

    // Convert metadata to YAML format
    YAML::Emitter out;
    out << YAML::BeginDoc;
    metadataToYAML(metadata, out);
    out << YAML::EndDoc;

    yaml = out.c_str();

    return yaml;
}
