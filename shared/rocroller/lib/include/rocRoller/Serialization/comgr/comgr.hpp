#pragma once

#include <rocRoller/Serialization/Base.hpp>
#include <rocRoller/Serialization/Containers.hpp>
#include <rocRoller/Serialization/HasTraits.hpp>

#include <rocRoller/Utilities/Error.hpp>

#include <amd_comgr/amd_comgr.h>

#include <cstddef>
#include <functional>
#include <memory>
#include <string>
#include <vector>

namespace rocRoller
{
    namespace Serialization
    {
        // Forward declaration
        struct COMGRInput;

        // Helper class to manage COMGR metadata nodes with RAII
        class MetadataNodeHandle
        {
        private:
            amd_comgr_metadata_node_t node;
            bool                      owned;

        public:
            MetadataNodeHandle()
                : node{0}
                , owned(false)
            {
            }

            explicit MetadataNodeHandle(amd_comgr_metadata_node_t n, bool take_ownership = true)
                : node(n)
                , owned(take_ownership)
            {
            }

            ~MetadataNodeHandle()
            {
                if(owned && node.handle != 0)
                {
                    amd_comgr_destroy_metadata(node);
                }
            }

            // Move constructor
            MetadataNodeHandle(MetadataNodeHandle&& other) noexcept
                : node(other.node)
                , owned(other.owned)
            {
                other.node.handle = 0;
                other.owned       = false;
            }

            // Move assignment
            MetadataNodeHandle& operator=(MetadataNodeHandle&& other) noexcept
            {
                if(this != &other)
                {
                    if(owned && node.handle != 0)
                    {
                        amd_comgr_destroy_metadata(node);
                    }
                    node              = other.node;
                    owned             = other.owned;
                    other.node.handle = 0;
                    other.owned       = false;
                }
                return *this;
            }

            // Delete copy constructor and assignment
            MetadataNodeHandle(const MetadataNodeHandle&) = delete;
            MetadataNodeHandle& operator=(const MetadataNodeHandle&) = delete;

            amd_comgr_metadata_node_t get() const
            {
                return node;
            }
            bool isValid() const
            {
                return node.handle != 0;
            }
        };

        // Helper functions for COMGR metadata conversion
        inline std::string getMetadataString(amd_comgr_metadata_node_t node)
        {
            size_t size   = 0;
            auto   status = amd_comgr_get_metadata_string(node, &size, nullptr);
            if(status != AMD_COMGR_STATUS_SUCCESS)
                return "";

            std::string result(size - 1, '\0'); // -1 to exclude null terminator
            status = amd_comgr_get_metadata_string(node, &size, result.data());
            if(status != AMD_COMGR_STATUS_SUCCESS)
                return "";

            return result;
        }

        inline amd_comgr_metadata_kind_t getMetadataKind(amd_comgr_metadata_node_t node)
        {
            amd_comgr_metadata_kind_t kind = AMD_COMGR_METADATA_KIND_NULL;
            amd_comgr_get_metadata_kind(node, &kind);
            return kind;
        }

        // Input handler for COMGR metadata
        struct COMGRInput
        {
            amd_comgr_metadata_node_t node;
            void*                     context;

            COMGRInput(amd_comgr_metadata_node_t n, void* c = nullptr)
                : node(n)
                , context(c)
            {
            }

            template <typename T>
            void mapRequired(const char* key, T& obj)
            {
                auto kind = getMetadataKind(node);
                AssertFatal(kind == AMD_COMGR_METADATA_KIND_MAP,
                            "mapRequired called on non-map metadata node");

                amd_comgr_metadata_node_t value_node = {0};
                auto status = amd_comgr_metadata_lookup(node, key, &value_node);
                if(status != AMD_COMGR_STATUS_SUCCESS)
                {
                    Throw<FatalError>("Required key '", key, "' not found in metadata map");
                }

                MetadataNodeHandle value_handle(value_node);
                input(value_handle.get(), obj);
            }

            template <typename T>
            void mapOptional(const char* key, T& obj)
            {
                auto kind = getMetadataKind(node);
                if(kind != AMD_COMGR_METADATA_KIND_MAP)
                    return;

                amd_comgr_metadata_node_t value_node = {0};
                auto status = amd_comgr_metadata_lookup(node, key, &value_node);
                if(status == AMD_COMGR_STATUS_SUCCESS)
                {
                    MetadataNodeHandle value_handle(value_node);
                    input(value_handle.get(), obj);
                }
            }

            // Input for mapped types
            template <typename T>
            requires(CMappedType<T, COMGRInput> || EmptyMappedType<T, COMGRInput>) void input(
                amd_comgr_metadata_node_t n, T& obj)
            {
                COMGRInput   subInput(n, context);
                EmptyContext ctx;
                MappingTraits<T, COMGRInput>::mapping(subInput, obj, ctx);
            }

            // Input for primitive types from string metadata
            template <typename T>
            void input(amd_comgr_metadata_node_t n, T& obj)
            {
                auto kind = getMetadataKind(n);
                if(kind == AMD_COMGR_METADATA_KIND_STRING)
                {
                    auto str = getMetadataString(n);
                    nodeInputHelper(str, obj);
                }
                else
                {
                    Throw<FatalError>("Expected string metadata for primitive type");
                }
            }

            // Specialization for string
            void input(amd_comgr_metadata_node_t n, std::string& obj)
            {
                auto kind = getMetadataKind(n);
                AssertFatal(kind == AMD_COMGR_METADATA_KIND_STRING, "Expected string metadata");
                obj = getMetadataString(n);
            }

            // Input for sequences
            template <SequenceType<COMGRInput> T>
            void input(amd_comgr_metadata_node_t n, T& obj)
            {
                auto kind = getMetadataKind(n);
                AssertFatal(kind == AMD_COMGR_METADATA_KIND_LIST,
                            "Expected list metadata for sequence");

                size_t count  = 0;
                auto   status = amd_comgr_get_metadata_list_size(n, &count);
                AssertFatal(status == AMD_COMGR_STATUS_SUCCESS, "Failed to get metadata list size");

                for(size_t i = 0; i < count; i++)
                {
                    amd_comgr_metadata_node_t elem_node = {0};
                    status = amd_comgr_index_list_metadata(n, i, &elem_node);
                    AssertFatal(status == AMD_COMGR_STATUS_SUCCESS, "Failed to get list element");

                    MetadataNodeHandle elem_handle(elem_node);
                    auto& value = SequenceTraits<T, COMGRInput>::element(*this, obj, i);
                    input(elem_handle.get(), value);
                }
            }

            // Input for custom mapping types
            template <CustomMappingType<COMGRInput> T>
            void input(amd_comgr_metadata_node_t n, T& obj)
            {
                auto kind = getMetadataKind(n);
                AssertFatal(kind == AMD_COMGR_METADATA_KIND_MAP,
                            "Expected map metadata for custom mapping");

                COMGRInput subInput(n, context);

                // Iterator callback for map entries
                struct IteratorContext
                {
                    COMGRInput* input;
                    T*          obj;
                };

                IteratorContext iter_ctx{&subInput, &obj};

                auto callback = [](amd_comgr_metadata_node_t key,
                                   amd_comgr_metadata_node_t value,
                                   void*                     user_data) -> amd_comgr_status_t {
                    auto* ctx     = static_cast<IteratorContext*>(user_data);
                    auto  key_str = getMetadataString(key);
                    CustomMappingTraits<T, COMGRInput>::inputOne(*ctx->input, key_str, *ctx->obj);
                    return AMD_COMGR_STATUS_SUCCESS;
                };

                auto status = amd_comgr_iterate_map_metadata(n, callback, &iter_ctx);
                AssertFatal(status == AMD_COMGR_STATUS_SUCCESS, "Failed to iterate metadata map");
            }

            // Input for scalar traits
            template <CHasScalarTraits T>
            void input(amd_comgr_metadata_node_t n, T& obj)
            {
                std::string stringVal = getMetadataString(n);
                ScalarTraits<T>::input(stringVal, obj);
            }

            constexpr bool outputting() const
            {
                return false;
            }

        private:
            // Helper for converting strings to primitive types
            template <typename T>
            void nodeInputHelper(const std::string& str, T& obj)
            {
                std::istringstream iss(str);
                iss >> obj;
            }
            /*
            void nodeInputHelper(const std::string& str, bool& val)
            {
                if(str == "true" || str == "1")
                    val = true;
                else if(str == "false" || str == "0")
                    val = false;
                else
                {
                    // Try to parse as integer
                    int intVal;
                    nodeInputHelper(str, intVal);
                    val = static_cast<bool>(intVal);
                }
            }*/

            void nodeInputHelper(const std::string& str, Half& val)
            {
                float floatVal;
                nodeInputHelper(str, floatVal);
                val = floatVal;
            }

            void nodeInputHelper(const std::string& str, BFloat16& val)
            {
                float floatVal;
                nodeInputHelper(str, floatVal);
                val.data = floatVal;
            }
        };

        // IOTraits specialization for COMGRInput
        template <>
        struct IOTraits<COMGRInput>
        {
            using IO = COMGRInput;

            template <typename T>
            static void mapRequired(IO& io, const char* key, T& obj)
            {
                io.mapRequired(key, obj);
            }

            template <typename T, typename Context>
            static void mapRequired(IO& io, const char* key, T& obj, Context& ctx)
            {
                io.mapRequired(key, obj);
            }

            template <typename T>
            static void mapOptional(IO& io, const char* key, T& obj)
            {
                io.mapOptional(key, obj);
            }

            template <typename T, typename Context>
            static void mapOptional(IO& io, const char* key, T& obj, Context& ctx)
            {
                io.mapOptional(key, obj);
            }

            static constexpr bool outputting(IO& io)
            {
                return io.outputting();
            }

            static void setError(IO& io, std::string const& msg)
            {
                throw std::runtime_error(msg);
            }

            static void setContext(IO& io, void* ctx)
            {
                io.context = ctx;
            }

            static void* getContext(IO& io)
            {
                return io.context;
            }

            template <typename T>
            static void enumCase(IO& io, T& member, const char* key, T value)
            {
                auto kind = getMetadataKind(io.node);
                if(kind == AMD_COMGR_METADATA_KIND_STRING)
                {
                    auto str = getMetadataString(io.node);
                    if(str == key)
                    {
                        member = value;
                    }
                }
            }
        };

    } // namespace Serialization
} // namespace rocRoller
