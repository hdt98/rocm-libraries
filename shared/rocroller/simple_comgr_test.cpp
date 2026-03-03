#include <amd_comgr/amd_comgr.h>
#include <string>
#include <vector>
#include <iostream>
#include <sstream>
#include <cassert>

// Helper macro to check for amd_comgr errors
#define COMGR_CHECK(cmd)                                                                      \
    do                                                                                        \
    {                                                                                         \
        amd_comgr_status_t e = cmd;                                                           \
        if(e != amd_comgr_status_t::AMD_COMGR_STATUS_SUCCESS)                                 \
        {                                                                                     \
            std::ostringstream msg;                                                           \
            char const*        statusMsg;                                                     \
            amd_comgr_status_string(e, &statusMsg);                                           \
            msg << "amd comgr failure at line " << __LINE__ << ": " << std::string(statusMsg) \
                << std::endl;                                                                 \
            std::cerr << msg.str() << std::endl;                                              \
        }                                                                                     \
    } while(0)

int main() {
  // const std::string isa = "amdgcn-amd-amdhsa--gfx950:sramecc+";
  const std::string isa = "amdgcn-amd-amdhsa--gfx90a:xnack+";

  const std::string src_name = "kernel.s";
  const std::string src = R"(
.amdgcn_target "amdgcn-amd-amdhsa--gfx90a:xnack+"
  )";

  std::vector<char> result;
  size_t            dataOutSize = 0;

  amd_comgr_data_t        assemblyData, execData;
  amd_comgr_data_set_t    assemblyDataSet, relocatableDataSet, execDataSet;
  amd_comgr_action_info_t dataAction;

  const char* codeGenOptions[]
      = {"-v", "-###",
         "-Xclang", "--amdhsa-code-object-version=5",
         "-mwavefrontsize64"};
  size_t codeGenOptionsCount = sizeof(codeGenOptions) / sizeof(codeGenOptions[0]);

  // Initialize Comgr data handles
  COMGR_CHECK(amd_comgr_create_data_set(&assemblyDataSet));
  COMGR_CHECK(amd_comgr_create_data(AMD_COMGR_DATA_KIND_SOURCE, &assemblyData));

  COMGR_CHECK(amd_comgr_set_data(assemblyData, src.size(), src.c_str()));
  COMGR_CHECK(amd_comgr_set_data_name(assemblyData, src_name.c_str()));
  COMGR_CHECK(amd_comgr_data_set_add(assemblyDataSet, assemblyData));

  COMGR_CHECK(amd_comgr_create_data_set(&relocatableDataSet));

  COMGR_CHECK(amd_comgr_create_data_set(&execDataSet));

  // Initialize Comgr action
  COMGR_CHECK(amd_comgr_create_action_info(&dataAction));
  COMGR_CHECK(amd_comgr_action_info_set_isa_name(dataAction, isa.c_str()));
  COMGR_CHECK(
      amd_comgr_action_info_set_option_list(dataAction, codeGenOptions, codeGenOptionsCount));

  // Assemble and link with Comgr
  COMGR_CHECK(amd_comgr_do_action(AMD_COMGR_ACTION_ASSEMBLE_SOURCE_TO_RELOCATABLE,
                                  dataAction,
                                  assemblyDataSet,
                                  relocatableDataSet));

  COMGR_CHECK(amd_comgr_do_action(AMD_COMGR_ACTION_LINK_RELOCATABLE_TO_EXECUTABLE,
                                  dataAction,
                                  relocatableDataSet,
                                  execDataSet));

  // Extract data from DataSet handle
  COMGR_CHECK(
      amd_comgr_action_data_get_data(execDataSet, AMD_COMGR_DATA_KIND_EXECUTABLE, 0, &execData));

  COMGR_CHECK(amd_comgr_get_data(execData, &dataOutSize, nullptr));
  assert(dataOutSize > 0 && "No compiled kernel data");
  result.resize(dataOutSize);
  COMGR_CHECK(amd_comgr_get_data(execData, &dataOutSize, result.data()));

  // Cleanup
  COMGR_CHECK(amd_comgr_destroy_data_set(assemblyDataSet));
  COMGR_CHECK(amd_comgr_destroy_data_set(relocatableDataSet));
  COMGR_CHECK(amd_comgr_destroy_data_set(execDataSet));
  COMGR_CHECK(amd_comgr_destroy_action_info(dataAction));
  COMGR_CHECK(amd_comgr_release_data(assemblyData));

  return 0;
}
