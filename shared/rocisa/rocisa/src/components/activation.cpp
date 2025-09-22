#include "code.hpp"
#include "container.hpp"
#include "enum.hpp"

#include <algorithm>
#include <iostream>
#include <stdexcept>
#include <string>
#include <unordered_map>
#include <vector>

////////////////////////////////////////////////////////////////////////////////
// How to add an activation
// 1. Add a new type in ActivationType
// 2. Create a new getXXXAssembly function in class Activation
// 3. Add if-else condition in generateAssembly in class Activation
// 4. Add if-else condition in generateInlineAssemblyBody in class
//    ActivationInline
//
// Helper function(s)
// 1. getRegAndInitAssembly
//    ```
//    getRegAndInitAssembly(<v for vgpr/ s for sgpr>,
//                          <False for reg pool/ True for tmp reg pool>,
//                          <size of reg>,
//                          <init value>,
//                          <key>,
//                          <comment>)
//    ```
//    Returns
//    1. sgprinf: The original checkOut return value
//    2. regInitStr: The init instruction string
//
//    Example,
//    ```
//    sgprinf, regInitStr = self.getRegAndInitAssembly('s', False, 1, \
//        "0x3f4c422a", "FloatGeluK0", "float gelu k0")
//    ```
//    this will generate ``regInitStr`` as
//    ```
//    s_mov_b32 sXX, "0x3f4c422a" // float16 max
//    ```
//    if the key "FloatGeluK0" is not found in sgprDict
// 2. class ActivationRegisterPool
//    A wrapper of RegisterPool. All the checkOut-ed registers will be
//    checkIn-ed at the end of the numBatches for loop. When ActivationType is
//    set to 'all', the registers will be checkIn-ed after activation's gwvw for
//    loop.
////////////////////////////////////////////////////////////////////////////////

////////////////////////////////////////////////////////////////////////////////
// This is the ActivationType class
// stringList:
//   This list stores the names of extra arguments, e.g.
//   y = (x > 0) ? x : x * alpha
// lookup:
//   This dict stores the supported activation types as keys and number of
//   arguments as values. Insert any new type before 'none' and 'all'. The
//   sequence of the table should match the enum in Activation.hpp.
//
// To add an activation type, see the instruction in Activation.cpp.
////////////////////////////////////////////////////////////////////////////////

namespace rocisa {
struct ActivationAvailable {
  bool half;
  bool single;
  bool double_;
  bool bfloat16;
  bool int8;
  bool int16;
  bool int32;

  ActivationAvailable(bool canHalf = false, bool canSingle = false,
                      bool canDouble = false, bool canBFloat16 = false,
                      bool canInt8 = false, bool canInt16 = false,
                      bool canInt32 = false)
      : half(canHalf), single(canSingle), double_(canDouble),
        bfloat16(canBFloat16), int8(canInt8), int16(canInt16), int32(canInt32) {
  }
};

struct ActivationTypeRegister {
  std::string name;
  bool isGradient;
  std::string extraArgs; // Assuming extraArgs is a string
  ActivationAvailable can;

  ActivationTypeRegister(const std::string &name, bool isGradient,
                         const std::string &extraArgs, bool canHalf = false,
                         bool canSingle = false, bool canDouble = false,
                         bool canBFloat16 = false, bool canInt8 = false,
                         bool canInt16 = false, bool canInt32 = false)
      : name(name), isGradient(isGradient), extraArgs(extraArgs),
        can(canHalf, canSingle, canDouble, canBFloat16, canInt8, canInt16,
            canInt32) {}

  bool typeAvailable(const DataType &dataType) const {
    if (dataType == DataType::Half && can.half) {
      return true;
    } else if (dataType == DataType::Float && can.single) {
      return true;
    } else if (dataType == DataType::Double && can.double_) {
      return true;
    } else if (dataType == DataType::BFloat16 && can.bfloat16) {
      return true;
    } else if (dataType == DataType::Int8 && can.int8) {
      return true;
    } else if (dataType == DataType::Int32 && can.int32) {
      return true;
    }
    return false;
  }
};

class ActivationType {
public:
  enum class Export { NORMAL = 0, GRADONLY = 1, BOTH = 2 };

  // Use bit mask to maintain the supported information.
  enum class SupportedBy { HIPBLASLT = 0b01, TENSILE = 0b10, ALL = 0b11 };

private:
  std::string value;
  static const std::vector<std::string> stringList;
  static const std::unordered_map<std::string, ActivationTypeRegister>
      lookupVeri;
  static const std::unordered_map<
      std::string, std::pair<ActivationTypeRegister, SupportedBy>>
      lookup;

public:
  ActivationType(const std::string &value);
  ActivationType(const ActivationType &other);

  static bool passActivation(bool isGradient, Export exportType);
  int getAdditionalArgNum(Export exportType = Export::NORMAL) const;

  // Check if the given components are supported by the configuration using a
  // bit mask. This function performs a bitwise AND operation between the config
  // and components to determine if the specified components are included in the
  // configuration.
  static bool fitSupported(SupportedBy config, SupportedBy components);

  std::vector<std::string>
  getAdditionalArgStringList(bool addPrefix = true) const;
  static int getEnumIndex(const std::string &enumStr);
  static std::vector<std::string>
  getEnumStrList(const DataType &dataType, SupportedBy configSupported,
                 bool includeNone = true, Export exportType = Export::NORMAL);

  std::string state() const;
  std::string toString() const;
  std::string toEnum() const;

  bool operator==(const std::string &other) const;
  bool operator==(const ActivationType &other) const;
  bool operator<(const std::string &other) const;
  bool operator<(const ActivationType &other) const;
};

// Static member definitions
// The order of the stringList must match the order of extraArgs in
// ActivationTypeRegister If you add a new activation with extra args, please
// update the stringList accordingly.
const std::vector<std::string> ActivationType::stringList = {"alpha", "beta",
                                                             "gamma", "delta"};

// Exp is only for verification. So we will not return exp in the supported
// list. Half,Single,Double,BFloat16, Int8, Int16, Int32
const std::unordered_map<std::string, ActivationTypeRegister>
    ActivationType::lookupVeri = {
        {'exp', ActivationTypeRegister("exp", true, "0", true, true, false,
                                       false, false, false, false)}};

// Note: The BFloat16 gemm uses Single type activations. The int8 gemm uses
// int32 type activations. Half,Single,Double,BFloat16, Int8, Int16, Int32
const std::unordered_map<
    std::string, std::pair<ActivationTypeRegister, ActivationType::SupportedBy>>
    ActivationType::lookup = {
        {"none",
         {ActivationTypeRegister("none", false, "0", true, true, true, true,
                                 true, true, true),
          ActivationType::SupportedBy::TENSILE |
              ActivationType::SupportedBy::HIPBLASLT}},
        {"abs",
         {ActivationTypeRegister("abs", false, "0", true, true, true, true,
                                 false, false, true),
          ActivationType::SupportedBy::TENSILE}},
        {"clippedrelu",
         {ActivationTypeRegister("clippedrelu", false, "2", true, true, true,
                                 false, false, false, true),
          ActivationType::SupportedBy::TENSILE}},
        {"gelu",
         {ActivationTypeRegister("gelu", false, "0", true, true, false, false,
                                 false, false, false),
          ActivationType::SupportedBy::TENSILE |
              ActivationType::SupportedBy::HIPBLASLT}},
        {"leakyrelu",
         {ActivationTypeRegister("leakyrelu", false, "1", true, true, true,
                                 false, false, false, true),
          ActivationType::SupportedBy::TENSILE}},
        {"relu",
         {ActivationTypeRegister("relu", false, "0", true, true, true, false,
                                 false, false, true),
          ActivationType::SupportedBy::TENSILE |
              ActivationType::SupportedBy::HIPBLASLT}},
        {"sigmoid",
         {ActivationTypeRegister("sigmoid", false, "0", true, true, false,
                                 false, false, false, false),
          ActivationType::SupportedBy::TENSILE}},
        {"tanh",
         {ActivationTypeRegister("tanh", false, "2", true, true, false, false,
                                 false, false, false),
          ActivationType::SupportedBy::TENSILE}},
        {"dgelu",
         {ActivationTypeRegister("dgelu", true, "0", false, true, false, false,
                                 false, false, false),
          ActivationType::SupportedBy::TENSILE |
              ActivationType::SupportedBy::HIPBLASLT}},
        {"geluscaling",
         {ActivationTypeRegister("geluscaling", false, "1", true, true, false,
                                 false, false, false, false),
          ActivationType::SupportedBy::TENSILE}},
        {"silu",
         {ActivationTypeRegister("silu", false, "0", true, true, false, false,
                                 false, false, false),
          ActivationType::SupportedBy::TENSILE |
              ActivationType::SupportedBy::HIPBLASLT}},
        {"swish",
         {ActivationTypeRegister("swish", false, "1", true, true, false, false,
                                 false, false, false),
          ActivationType::SupportedBy::TENSILE}},
        {"clamp",
         {ActivationTypeRegister("clamp", false, "2", true, true, true, false,
                                 false, false, true),
          ActivationType::SupportedBy::TENSILE |
              ActivationType::SupportedBy::HIPBLASLT}},
        {"hipblaslt_all",
         {ActivationTypeRegister("hipblaslt_all", false, "0"),
          ActivationType::SupportedBy::HIPBLASLT}},
        {"all",
         {ActivationTypeRegister("all", false, "0"),
          ActivationType::SupportedBy::TENSILE |
              ActivationType::SupportedBy::HIPBLASLT}}};

ActivationType::ActivationType(const std::string &value) {
  if (value.empty()) {
    throw std::runtime_error("Activation type cannot be empty");
  }
  std::string strValue = value;
  std::transform(strValue.begin(), strValue.end(), strValue.begin(), ::tolower);
  if (lookup.find(strValue) != lookup.end()) {
    this->value = strValue;
  } else if (lookupVeri.find(strValue) != lookupVeri.end()) {
    this->value = strValue;
  } else {
    throw std::runtime_error("Unrecognized activation type " + value);
  }
}

ActivationType::ActivationType(const ActivationType &other)
    : value(other.value) {}

bool ActivationType::passActivation(bool isGradient, Export exportType) {
  if (exportType == Export::NORMAL) {
    return isGradient;
  } else if (exportType == Export::GRADONLY) {
    return !isGradient;
  } else if (exportType == Export::BOTH) {
    return false;
  }
  return false; // Should not reach here, but added to avoid compiler warnings
}

int ActivationType::getAdditionalArgNum(Export exportType) const {
  if (this->value == "all" || this->value == "hipblaslt_all") {
    int maxArgNum = 0;
    for (const auto &pair : lookup) {
      const ActivationTypeRegister &activationInst = pair.second.first;
      if (ActivationType::passActivation(activationInst.isGradient,
                                         exportType)) {
        continue;
      }
      maxArgNum = std::max(maxArgNum, std::stoi(activationInst.extraArgs));
    }
    return maxArgNum;
  } else if (lookup.find(this->value) != lookup.end()) {
    return std::stoi(lookup.at(this->value).first.extraArgs);
  }
  return 0;
}

// Check if the given components are supported by the configuration using a bit
// mask. This function performs a bitwise AND operation between the config and
// components to determine if the specified components are included in the
// configuration.
bool ActivationType::fitSupported(SupportedBy config, SupportedBy components) {
  return (static_cast<int>(config) & static_cast<int>(components)) != 0;
}

std::vector<std::string>
ActivationType::getAdditionalArgStringList(bool addPrefix = true) const {
  std::vector<std::string> list;
  int argNum = this->getAdditionalArgNum();
  for (int i = 0; i < argNum; ++i) {
    if (addPrefix) {
      std::string capitalized = stringList[i];
      if (!capitalized.empty()) {
        capitalized[0] = std::toupper(capitalized[0]);
      }
      list.push_back("activation" + capitalized);
    } else {
      list.push_back(stringList[i]);
    }
  }
  return list;
}

int ActivationType::getEnumIndex(const std::string &enumStr) {
  std::string strValue = enumStr;
  std::transform(strValue.begin(), strValue.end(), strValue.begin(), ::tolower);
  auto it = lookup.find(strValue);
  if (it != lookup.end()) {
    return std::distance(lookup.begin(), it);
  }
  throw std::runtime_error("Activation type not found: " + strValue);
}

std::vector<std::string>
ActivationType::getEnumStrList(const DataType &dataType,
                               SupportedBy configSupported, bool includeNone,
                               Export exportType) {
  std::vector<std::string> enumList;
  for (const auto &pair : lookup) {
    const std::string &key = pair.first;
    const ActivationTypeRegister &activationInst = pair.second.first;
    SupportedBy components = pair.second.second;

    if (ActivationType::passActivation(activationInst.isGradient, exportType)) {
      continue;
    }

    if (((key != "none") || includeNone) &&
        (key != "all" && key != "hipblaslt_all")) {
      if (activationInst.typeAvailable(dataType) &&
          fitSupported(configSupported, components)) {
        enumList.push_back(key);
      }
    }
  }

  if (enumList.empty()) {
    std::cout << "No available activation for this data type "
              << rocisa::toString(dataType) << ".\n";
  }

  return enumList;
}

std::string ActivationType::state() const {
  std::string result = this->value;
  if (!result.empty()) {
    result[0] = std::toupper(result[0]);
  }
  return result;
}

std::string ActivationType::toString() const {
  std::string result = this->value;
  if (!result.empty()) {
    result[0] = std::toupper(result[0]);
  }
  return result;
}

bool ActivationType::operator==(const std::string &other) const {
  std::string otherLower = other;
  std::transform(otherLower.begin(), otherLower.end(), otherLower.begin(),
                 ::tolower);
  return this->value == otherLower;
}

bool ActivationType::operator==(const ActivationType &other) const {
  return this->value == other.value;
}

bool ActivationType::operator<(const std::string &other) const {
  std::string otherLower = other;
  std::transform(otherLower.begin(), otherLower.end(), otherLower.begin(),
                 ::tolower);
  return this->value < otherLower;
}

bool ActivationType::operator<(const ActivationType &other) const {
  return this->value < other.value;
}

std::string ActivationType::toEnum() const {
  std::string result = this->value;
  if (!result.empty()) {
    result[0] = std::toupper(result[0]);
  }
  return result;
}

struct ActCacheInfo {
  bool usePK;
  bool saturateI8;
  bool enableGuard;
  bool isAlt;
  std::string prefix;
  std::vector<std::vector<std::shared_ptr<RegisterContainer>>> vgprIdxList;
  std::shared_ptr<Module> module;
  int vgprCounter;
  int sgprCounter;

  bool isSame(bool usePK, bool saturateI8, bool enableGuard, bool isAlt,
              const std::string &prefix) const {
    return (this->usePK == usePK) && (this->saturateI8 == saturateI8) &&
           (this->enableGuard == enableGuard) && (this->prefix == prefix) &&
           (this->isAlt == isAlt);
  }
};

union ActivationMagicNumber {
  uint32_t hex;
  float f;

  constexpr ActivationMagicNumber(uint32_t hexValue) : hex(hexValue) {}
};

const std::unordered_map<std::string, ActivationMagicNumber>
    ActivationMagicNumbers = {
        {"FloatGeluK0", ActivationMagicNumber(0x3f4c422a)},
        {"FloatGeluK1", ActivationMagicNumber(0x3d372713)},
        {"Float16GeluK1", ActivationMagicNumber(0x29b9)},
        {"FloatDGeluK0", ActivationMagicNumber(0x3d5b33b3)},
        {"FloatDGeluK1", ActivationMagicNumber(0x3ecc4220)},
        {"FloatDGeluK2", ActivationMagicNumber(0x3d12220c)},
        {"FloatDGeluK3", ActivationMagicNumber(0x3f4c4231)}};

class ActivationModule {
private:
  // Counter
  int vgprCounter;
  int sgprCounter;

  // Properties
  bool usePK;
  bool saturateI8;
  std::string vgprPrefixFormat;
  bool needCombine;
  bool enableGuard;
  bool isAlt;

  // Cache
  std::unordered_map<std::string,
                     std::unordered_map<char, std::vector<ActCacheInfo>>>
      cacheDict;
  bool useCache;
  int labelCounter;

  // Helper methods
  void resetGprCounter();
  int getVgpr(int num);
  int getSgpr(int num);
  std::string vgprPrefix(int idx);
  std::string vgprPrefix(int idx, int size);

  // Activation function implementations
  std::shared_ptr<Module> getAbsModule(const DataType &cDataType, int vgprIn,
                                       int vgprOut);
  std::shared_ptr<Module>
  getClippedReluModule(const DataType &cDataType, int vgprIn, int vgprOut,
                       const std::string &activationAlpha,
                       const std::string &activationBeta);
  std::shared_ptr<Module> getExpModule(const DataType &cDataType, int vgprIn,
                                       int vgprOut);
  std::shared_ptr<Module>
  getGeluModule(const DataType &cDataType, int vgprIn, int vgprOut,
                const std::string &activationAlpha = "");
  std::shared_ptr<Module>
  getLeakyReluModule(const DataType &cDataType, int vgprIn, int vgprOut,
                     const std::string &activationAlpha);
  std::shared_ptr<Module> getReluModule(const DataType &cDataType, int vgprIn,
                                        int vgprOut);
  std::shared_ptr<Module> getSigmoidModule(const DataType &cDataType,
                                           int vgprIn, int vgprOut);
  std::shared_ptr<Module> getTanhModule(const DataType &cDataType, int vgprIn,
                                        int vgprOut,
                                        const std::string &activationAlpha,
                                        const std::string &activationBeta);
  std::shared_ptr<Module> getDGeluModule(const DataType &cDataType, int vgprIn,
                                         int vgprOut);
  std::shared_ptr<Module> getSiluModule(const DataType &cDataType, int vgprIn,
                                        int vgprOut);
  std::shared_ptr<Module> getSwishModule(const DataType &cDataType, int vgprIn,
                                         int vgprOut,
                                         const std::string &activationAlpha);
  std::shared_ptr<Module> getClampModule(const DataType &cDataType, int vgprIn,
                                         int vgprOut,
                                         const std::string &activationAlpha,
                                         const std::string &activationBeta);

  // Cache functions
  void createCache(const DataType &cDataType, const std::string &activationType,
                   int vgprIn, int vgprOut, std::shared_ptr<Module> module);
  std::shared_ptr<Module> getCache(const DataType &cDataType,
                                   const std::string &activationType,
                                   int vgprIn, int vgprOut);

  // Helper functions
  std::shared_ptr<Module> postProcess(const DataType &cDataType,
                                      std::shared_ptr<Module> module);
  std::shared_ptr<Module> assignGpr(std::shared_ptr<Module> module, int vgprIdx,
                                    int sgprIdx);

public:
  ActivationModule();

  // Public functions
  std::shared_ptr<Module> getModule(const DataType &cDataType,
                                    const std::string &activationType,
                                    int vgprIn, int vgprOut);
  std::unordered_map<std::string, std::unordered_map<std::string, int>>
  getAllGprUsage(
      const DataType &cDataType, const std::string &actType,
      ActivationType::Export exportType = ActivationType::Export::NORMAL);

  // Setters
  void setUsePK(bool usePK);
  void setSaturationForInt8(bool sat);
  void setVgprPrefixFormat(const std::string &formatting);
  void setUseCache(bool cache);
  void setGuard(bool guard);
  void setAlt(bool alt);
};

ActivationModule::ActivationModule() {
  // Counter
  this->vgprCounter = 0;
  this->sgprCounter = 0;
  // Properties
  this->usePK = true;
  this->saturateI8 = false;
  this->vgprPrefixFormat = "";
  // We only need to run CombineInstructions if the called module calls another
  // module inside.
  this->needCombine = false;
  // Cache
  this->cacheDict = {};
  this->useCache = false;
  this->labelCounter = 0;

  this->enableGuard = false;
  this->isAlt = false;
}

#Public function
std::shared_ptr<Module>
ActivationModule::getModule(const DataType &cDataType,
                            const std::string &activationType, int vgprIn,
                            int vgprOut) {
  if (useCache) {
    std::shared_ptr<Module> cachedModule =
        getCache(cDataType, activationType, vgprIn, vgprOut);
    if (!cachedModule->isEmpty()) { // Assuming Module has isEmpty() method
      return cachedModule;
    }
  }

  std::shared_ptr<Module> module;
  needCombine = false;
  resetGprCounter();

  if (activationType == "abs") {
    module = getAbsModule(cDataType, vgprIn, vgprOut);
  } else if (activationType == "clippedrelu") {
    module = getClippedReluModule(cDataType, vgprIn, vgprOut, "activationAlpha",
                                  "activationBeta");
  } else if (activationType == "exp") {
    module = getExpModule(cDataType, vgprIn, vgprOut);
  } else if (activationType == "gelu") {
    module = getGeluModule(cDataType, vgprIn, vgprOut);
  } else if (activationType == "geluscaling") {
    module = getGeluModule(cDataType, vgprIn, vgprOut, "activationAlpha");
  } else if (activationType == "leakyrelu") {
    module = getLeakyReluModule(cDataType, vgprIn, vgprOut, "activationAlpha");
  } else if (activationType == "relu") {
    module = getReluModule(cDataType, vgprIn, vgprOut);
  } else if (activationType == "sigmoid") {
    module = getSigmoidModule(cDataType, vgprIn, vgprOut);
  } else if (activationType == "tanh") {
    module = getTanhModule(cDataType, vgprIn, vgprOut, "activationAlpha",
                           "activationBeta");
  } else if (activationType == "dgelu") {
    module = getDGeluModule(cDataType, vgprIn, vgprOut);
  } else if (activationType == "silu") {
    module = getSiluModule(cDataType, vgprIn, vgprOut);
  } else if (activationType == "swish") {
    module = getSwishModule(cDataType, vgprIn, vgprOut, "activationAlpha");
  } else if (activationType == "clamp") {
    module = getClampModule(cDataType, vgprIn, vgprOut, "activationAlpha",
                            "activationBeta");
  } else if (activationType == "none") {
    return std::make_shared<Module>("No activation");
  } else {
    return std::make_shared<Module>(activationType + " not implemented");
  }

  module = postProcess(cDataType, module);
  // Only create cache when In != Out else will cause an error
  if (useCache && (vgprIn != vgprOut)) {
    createCache(cDataType, activationType, vgprIn, vgprOut, module);
  }
  return module;
}

std::unordered_map<std::string, std::unordered_map<std::string, int>>
ActivationModule::getAllGprUsage(const DataType &cDataType,
                                 const std::string &actType,
                                 ActivationType::Export exportType) {
  std::unordered_map<std::string, std::unordered_map<std::string, int>> usage;
  std::vector<std::string> enumList;

  if (actType == "all") {
    enumList = ActivationType::getEnumStrList(
        cDataType, ActivationType::SupportedBy::ALL, true, exportType);
  } else if (actType == "hipblaslt_all") {
    enumList = ActivationType::getEnumStrList(
        cDataType, ActivationType::SupportedBy::HIPBLASLT, true, exportType);
  } else {
    enumList = {actType};
  }

  for (const std::string &enumStr : enumList) {
    getModule(cDataType, enumStr, 0, 1); // dummy vgpr
    usage[enumStr] = {{"vgpr", vgprCounter}, {"sgpr", sgprCounter}};
  }

  return usage;
}

std::shared_ptr<Module>
ActivationModule::postProcess(const DataType &cDataType,
                              std::shared_ptr<Module> module) {
  if (needCombine) {
    CombineInstructions(module);
  }
  module = ConvertCoeffToHex(module, cDataType, usePK);
  return module;
}

std::shared_ptr<Module>
ActivationModule::assignGpr(std::shared_ptr<Module> module, int vgprIdx,
                            int sgprIdx) {
  std::vector<std::string> patternPrefix = {"v", "s"};
  std::vector<int> gprIdx = {vgprIdx, sgprIdx};

  for (size_t idx = 0; idx < patternPrefix.size(); ++idx) {
    module = HolderToGpr(module, gprIdx[idx], patternPrefix[idx]);
  }
  return module;
}

void ActivationModule::setUsePK(bool usePK) { this->usePK = usePK; }

void ActivationModule::setSaturationForInt8(bool sat) {
  this->saturateI8 = sat;
}

void ActivationModule::setVgprPrefixFormat(const std::string &formatting) {
  this->vgprPrefixFormat = formatting;
}

void ActivationModule::setUseCache(bool cache) { this->useCache = cache; }

void ActivationModule::setGuard(bool guard) { this->enableGuard = guard; }

void ActivationModule::setAlt(bool alt) { this->isAlt = alt; }

////////////////////////////////////////////////////////////////////////////////
////////////////////////////////////////////////////////////////////////////////
///
///   Internal Helper Functions
///
////////////////////////////////////////////////////////////////////////////////
////////////////////////////////////////////////////////////////////////////////

void ActivationModule::resetGprCounter() {
  this->vgprCounter = 0;
  this->sgprCounter = 0;
}

int ActivationModule::getVgpr(int num) {
  int value = this->vgprCounter;
  this->vgprCounter += num;
  return value;
}

int ActivationModule::getSgpr(int num) {
  int value = this->sgprCounter;
  this->sgprCounter += num;
  return value;
}

std::string ActivationModule::vgprPrefix(int idx) {
  if (!vgprPrefixFormat.empty()) {
    char buffer[256];
    snprintf(buffer, sizeof(buffer), vgprPrefixFormat.c_str(), idx);
    return std::string(buffer);
  }
  return std::to_string(idx);
}

std::string ActivationModule::vgprPrefix(int idx, int size) {
  std::string vgprStr = vgprPrefix(idx);
  return vgprStr + ":" + std::to_string(size);
}

////////////////////////////////////////////////////////////////////////////////
////////////////////////////////////////////////////////////////////////////////
///
///   Activation Functions
///
////////////////////////////////////////////////////////////////////////////////
////////////////////////////////////////////////////////////////////////////////

std::shared_ptr<Module>
ActivationModule::getAbsModule(const DataType &cDataType, int vgprIn,
                               int vgprOut) {
  std::shared_ptr<Module> module = std::make_shared<Module>("Abs");

  if (cDataType == DataType::Half || cDataType == DataType::BFloat16) {
    std::string absMagic = usePK ? "0x7fff7fff" : "0x7fff";
            module->addT<VAndB32>(vgprPrefix(vgprOut), absMagic, vgprPrefix(vgprIn), "Remove sign bit"));
  } else if (cDataType == DataType::Float) {
    module->addT<VAndB32>(vgprPrefix(vgprOut), "0x7fffffff", vgprPrefix(vgprIn),
                          "Remove sign bit");
  } else if (cDataType == DataType::Double) {
    module->addT<VAndB32>(vgprPrefix(vgprOut + 1), "0x7fffffff",
                          vgprPrefix(vgprIn + 1), "Remove sign bit");
  } else if (cDataType == DataType::Int32) {
    int vgprTemp = this->getVgpr(1);
    module->addT<VSubI32>(vgpr(Holder(idx = vgprTemp)), 0, vgprPrefix(vgprIn),
                          "x2 = -x");
    if (this->saturateI8) {
      int vgprTemp2 = this->getVgpr(1);
      module->addT<VMovB32>(vgpr(Holder(idx = vgprTemp2)), 127, "value = 127");
      module->addT<VMed3I32>(
          vgprPrefix(vgprOut), vgprPrefix(vgprIn), vgpr(Holder(idx = vgprTemp)),
          vgpr(Holder(idx = vgprTemp2)), "y = min(127, max(x, x2))");
    } else {
      module->addT<VMaxI32>(vgprPrefix(vgprOut), vgpr(Holder(idx = vgprTemp)),
                            vgprPrefix(vgprIn), "y = max(x, x2)");
    }
  } else {
    throw std::runtime_error("Unsupported data type " +
                             rocisa::toString(cDataType) + ".");
  }

  return module;
}

std::shared_ptr<Module> ActivationModule::getClippedReluModule(
    const DataType &cDataType, int vgprIn, int vgprOut,
    const std::string &activationAlpha, const std::string &activationBeta) {
  std::shared_ptr<Module> module = std::make_shared<Module>("ClippedRelu");

  if (cDataType == DataType::Half || cDataType == DataType::BFloat16) {
    for (int i = 0; i < 2; ++i) {
      SelectBit select_bit = (i == 0) ? SelectBit::WORD_0 : SelectBit::WORD_1;
      module->addT<VCmpGTF16>(
          VCC(), vgprPrefix(vgprIn), "sgpr(" + activationAlpha + ")",
          SDWAModifiers(SelectBit::DWORD, UnusedBit::UNUSED_PAD, select_bit,
                        SelectBit::WORD_0),
          "x > alpha?");
      module->addT<VMinF16>(vgprPrefix(vgprOut), "sgpr(" + activationBeta + ")",
                            vgprPrefix(vgprIn),
                            SDWAModifiers(select_bit,
                                          UnusedBit::UNUSED_PRESERVE,
                                          select_bit, select_bit),
                            "min(x, beta)");
      module->addT<VCndMaskB32>(vgprPrefix(vgprOut), "0.0", vgprPrefix(vgprOut),
                                SDWAModifiers(select_bit,
                                              UnusedBit::UNUSED_PRESERVE,
                                              select_bit, select_bit),
                                "set x to 0 if <= alpha");
    }
    module->addT<SNop>(0, "1 wait states"); // workaround for emulator
  } else if (cDataType == DataType::Float) {
    module->addT<VCmpGTF32>(VCC(), vgprPrefix(vgprIn),
                            "sgpr(" + activationAlpha + ")", "x > alpha ?");
    module->addT<VMinF32>(vgprPrefix(vgprIn), "sgpr(" + activationBeta + ")",
                          vgprPrefix(vgprIn), "min(x, beta)");
    module->addT<VCndMaskB32>(vgprPrefix(vgprIn), "0.0", vgprPrefix(vgprIn),
                              "set x to 0 if <= alpha");
  } else if (cDataType == DataType::Double) {
    module->addT<VCmpGTF64>(VCC(), vgprPrefix(vgprIn, 2),
                            "sgpr(" + activationAlpha + ", 2)", "x > alpha ?");
    module->addT<VMinF64>(vgprPrefix(vgprIn, 2),
                          "sgpr(" + activationBeta + ", 2)",
                          vgprPrefix(vgprIn, 2), "min(x, beta)");
    module->addT<VCndMaskB32>(vgprPrefix(vgprIn), "0", vgprPrefix(vgprIn),
                              "set x to 0 if <= alpha");
    module->addT<VCndMaskB32>(vgprPrefix(vgprIn + 1), "0",
                              vgprPrefix(vgprIn + 1), "set x to 0 if <= alpha");
  } else if (cDataType == DataType::Int32) {
    module->addT<VCmpGTI32>(VCC(), vgprPrefix(vgprIn),
                            "sgpr(" + activationAlpha + ")", "x > alpha ?");
    module->addT<VMinI32>(vgprPrefix(vgprIn), "sgpr(" + activationBeta + ")",
                          vgprPrefix(vgprIn), "min(x, beta)");
    module->addT<VCndMaskB32>(vgprPrefix(vgprIn), "0.0", vgprPrefix(vgprIn),
                              "set x to 0 if <= alpha");
  } else {
    throw std::runtime_error("Unsupported data type " +
                             rocisa::toString(cDataType) + ".");
  }

  return module;
}

std::shared_ptr<Module>
ActivationModule::getExpModule(const DataType &cDataType, int vgprIn,
                               int vgprOut) {
  std::shared_ptr<Module> module = std::make_shared<Module>("Exp");

  if (cDataType == DataType::Half) {
    int sgprMagic = getSgpr(1);
    float logE2 = std::log(std::exp(1.0)) / std::log(2.0); // log2(e)
    module->addT<SMovB32>(sgpr(Holder(sgprMagic)), logE2, "exp magic");

    if (usePK) {
      module->addT<VMulPKF16>(vgprPrefix(vgprOut), sgpr(Holder(sgprMagic)),
                              vgprPrefix(vgprIn), "exp step 1");
      for (int i = 0; i < 2; ++i) {
        SelectBit select_bit = (i == 0) ? SelectBit::WORD_0 : SelectBit::WORD_1;
        module->addT<VExpF16>(
            vgprPrefix(vgprOut), vgprPrefix(vgprOut),
            SDWAModifiers(select_bit, UnusedBit::UNUSED_PRESERVE, select_bit),
            "exp step 2");
        // Add wait state if needed based on architecture capabilities
        if (rocIsa::getInstance().getArchCaps()["TransOpWait"])
          module->addT<SNop>(0, "1 wait states");
      }
    } else {
      module->addT<VMulF16>(vgprPrefix(vgprOut), sgpr(Holder(sgprMagic)),
                            vgprPrefix(vgprIn), "exp step 1");
      module->addT<VExpF16>(vgprPrefix(vgprOut), vgprPrefix(vgprOut),
                            "exp step 2");
      // Add wait state if needed based on architecture capabilities
      if (rocIsa::getInstance().getArchCaps()["TransOpWait"])
        module->addT<SNop>(0, "1 wait states");
    }
  } else if (cDataType == DataType::Float) {
    float logE2 = std::log(std::exp(1.0)) / std::log(2.0); // log2(e)
    module->addT<VMulF32>(vgprPrefix(vgprOut), logE2, vgprPrefix(vgprIn),
                          "exp step 1");
    module->addT<VExpF32>(vgprPrefix(vgprOut), vgprPrefix(vgprOut),
                          "exp step 2");
    // Add wait state if needed based on architecture capabilities
    if (rocIsa::getInstance().getArchCaps()["TransOpWait"])
      module->addT<SNop>(0, "1 wait states");
  } else {
    throw std::runtime_error("Unsupported data type " +
                             rocisa::toString(cDataType) + ".");
  }

  return module;
}

std::shared_ptr<Module>
ActivationModule::getGeluModule(const DataType &cDataType, int vgprIn,
                                int vgprOut,
                                const std::string &activationAlpha) {
  needCombine = true;
  std::shared_ptr<Module> module = std::make_shared<Module>("Gelu");

  // Gelu(x) = 0.5 * x * (1 + tanh(k0 * x * (1 + k1 * x * x)))
  if (cDataType == DataType::Half) {
    auto k1Magic = ActivationMagicNumbers.at("Float16GeluK1");
    std::string flt16GeluK1Str = HexToStr(cDataType, usePK, k1Magic.hex);
    int sgprMagicK1 = getSgpr(1);
    int sgprPKLiteral = getSgpr(1);

    module->addT<SMovB32>(sgpr(Holder(sgprMagicK1)), flt16GeluK1Str,
                          "Float16GeluK1");

    auto k0Magic = ActivationMagicNumbers.at("FloatGeluK0");
    module->addT<SMovB32>(sgpr(Holder(sgprPKLiteral)), k0Magic.f,
                          "FloatGeluK0");

    int vgprTemp = getVgpr(1);

    if (usePK) {
      module->addT<VMulPKF16>(vgpr(Holder(vgprTemp)), vgprPrefix(vgprIn),
                              vgprPrefix(vgprIn), "x * x");
      module->addT<VFmaPKF16>(vgpr(Holder(vgprTemp)), vgpr(Holder(vgprTemp)),
                              sgpr(Holder(sgprMagicK1)), "1.0",
                              VOP3PModifiers().setOpSelHi({1, 1, 0, 1}),
                              "x^2 * k1 + 1");
      module->addT<VMulPKF16>(vgpr(Holder(vgprTemp)), vgprPrefix(vgprIn),
                              vgpr(Holder(vgprTemp)), "x * (x^2 * k1 + 1)");
      module->addT<VMulPKF16>(
          vgpr(Holder(vgprTemp)), sgpr(Holder(sgprPKLiteral)),
          vgpr(Holder(vgprTemp)), "k0 * x * (x^2 * k1 + 1)");

      auto tanhModule = getTanhModule(cDataType, vgprTemp, vgprTemp, "", "");
      module->addModuleAsFlatItems(tanhModule);

      module->addT<VAddPKF16>(
          vgpr(Holder(vgprTemp)), "1.0", vgpr(Holder(vgprTemp)),
          VOP3PModifiers().setOpSelHi({0, 1, 1}), "1 + tanh(...)");
      module->addT<VMulPKF16>(vgpr(Holder(vgprTemp)), vgprPrefix(vgprIn),
                              vgpr(Holder(vgprTemp)), "x * (1 + tanh(...))");

      if (activationAlpha.empty()) {
        module->addT<VMulPKF16>(vgprPrefix(vgprOut), "0.5",
                                vgpr(Holder(vgprTemp)),
                                VOP3PModifiers().setOpSelHi({0, 1, 1}),
                                "0.5 * x * (1 + tanh(...))");
      } else {
        module->addT<VMulPKF16>(vgpr(Holder(vgprTemp)), "0.5",
                                vgpr(Holder(vgprTemp)),
                                VOP3PModifiers().setOpSelHi({0, 1, 1}),
                                "0.5 * x * (1 + tanh(...))");
        module->addT<VMulPKF16>(
            vgprPrefix(vgprOut), "sgpr(" + activationAlpha + ")",
            vgpr(Holder(vgprTemp)), VOP3PModifiers().setOpSelHi({0, 1, 1}),
            "0.5 * x * (1 + tanh(...)) * scale");
      }
    } else {
      module->addT<VMulF16>(vgpr(Holder(vgprTemp)), vgprPrefix(vgprIn),
                            vgprPrefix(vgprIn), "x * x");
      module->addT<VFmaF16>(vgpr(Holder(vgprTemp)), vgpr(Holder(vgprTemp)),
                            sgpr(Holder(sgprMagicK1)), "1.0", "x^2 * k1 + 1");
      module->addT<VMulF16>(vgpr(Holder(vgprTemp)), vgprPrefix(vgprIn),
                            vgpr(Holder(vgprTemp)), "x * (x^2 * k1 + 1)");
      module->addT<VMulF16>(vgpr(Holder(vgprTemp)), sgpr(Holder(sgprPKLiteral)),
                            vgpr(Holder(vgprTemp)), "k0 * x * (x^2 * k1 + 1)");

      auto tanhModule = getTanhModule(cDataType, vgprTemp, vgprTemp, "", "");
      module->addModuleAsFlatItems(tanhModule);

      module->addT<VAddF16>(vgpr(Holder(vgprTemp)), "1.0",
                            vgpr(Holder(vgprTemp)), "1 + tanh(...)");
      module->addT<VMulF16>(vgpr(Holder(vgprTemp)), vgprPrefix(vgprIn),
                            vgpr(Holder(vgprTemp)), "x * (1 + tanh(...))");

      if (activationAlpha.empty()) {
        module->addT<VMulF16>(vgprPrefix(vgprOut), "0.5",
                              vgpr(Holder(vgprTemp)),
                              "0.5 * x * (1 + tanh(...))");
      } else {
        module->addT<VMulF16>(vgpr(Holder(vgprTemp)), "0.5",
                              vgpr(Holder(vgprTemp)),
                              "0.5 * x * (1 + tanh(...))");
        module->addT<VMulF16>(
            vgprPrefix(vgprOut), "sgpr(" + activationAlpha + ")",
            vgpr(Holder(vgprTemp)), "0.5 * x * (1 + tanh(...)) * scale");
      }
    }
  } else if (cDataType == DataType::Float) {
    int vgprTemp = getVgpr(1);
    auto k1Magic = ActivationMagicNumbers.at("FloatGeluK1");
    std::string flt16GeluK1Str = HexToStr(cDataType, usePK, k1Magic.hex);

    module->addT<VMulF32>(vgpr(Holder(vgprTemp)), flt16GeluK1Str,
                          vgprPrefix(vgprIn), "k1 * x");
    module->addT<VFmaF32>(vgpr(Holder(vgprTemp)), vgprPrefix(vgprIn),
                          vgpr(Holder(vgprTemp)), "1.0", "1 + (k1 * x * x)");
    module->addT<VMulF32>(vgpr(Holder(vgprTemp)), vgprPrefix(vgprIn),
                          vgpr(Holder(vgprTemp)), "x * (1 + k1 * x * x)");

    auto k0Magic = ActivationMagicNumbers.at("FloatGeluK0");
    module->addT<VMulF32>(vgpr(Holder(vgprTemp)), k0Magic.f,
                          vgpr(Holder(vgprTemp)), "k0 * x * (x^2 * k1 + 1)");

    auto tanhModule = getTanhModule(cDataType, vgprTemp, vgprTemp, "", "");
    module->addModuleAsFlatItems(tanhModule);

    module->addT<VAddF32>(vgpr(Holder(vgprTemp)), "1.0", vgpr(Holder(vgprTemp)),
                          "1 + tanh(...)");
    module->addT<VMulF32>(vgpr(Holder(vgprTemp)), vgprPrefix(vgprIn),
                          vgpr(Holder(vgprTemp)), "x * (1 + tanh(...))");

    if (activationAlpha.empty()) {
      module->addT<VMulF32>(vgprPrefix(vgprOut), "0.5", vgpr(Holder(vgprTemp)),
                            "0.5 * x * (1 + tanh(...))");
    } else {
      module->addT<VMulF32>(vgpr(Holder(vgprTemp)), "0.5",
                            vgpr(Holder(vgprTemp)),
                            "0.5 * x * (1 + tanh(...))");
      module->addT<VMulF32>(
          vgprPrefix(vgprOut), "sgpr(" + activationAlpha + ")",
          vgpr(Holder(vgprTemp)), "0.5 * x * (1 + tanh(...)) * scale");
    }
  } else {
    throw std::runtime_error("Unsupported data type " +
                             rocisa::toString(cDataType) + ".");
  }

  return module;
}

    def getLeakyReluModule(self, cDataType, vgprIn, vgprOut, activationAlpha):
        module = Module("LeakyRelu")
        if cDataType.isHalf():
            vgprTemp = self.getVgpr(1)
            module.add(VMulPKF16(dst=vgpr(Holder(idx=vgprTemp)), src0=sgpr(activationAlpha), src1=self.vgprPrefix(vgprIn), comment="tmp = x * alpha"))
            for i in range(0, 2):
                select_bit = SelectBit.WORD_0 if i == 0 else SelectBit.WORD_1
                module.add(VCmpGEF16(dst=VCC(), src0=self.vgprPrefix(vgprIn), src1=0.0, \
                                     sdwa=SDWAModifiers(src0_sel=select_bit, src1_sel=SelectBit.WORD_0), \
                                     comment="x > 0 ?"))
                module.add(VCndMaskB32(dst=self.vgprPrefix(vgprOut), src0=vgpr(Holder(idx=vgprTemp)), src1=self.vgprPrefix(vgprIn), \
                                       sdwa=SDWAModifiers(dst_sel=select_bit, dst_unused=UnusedBit.UNUSED_PRESERVE, \
                                                          src0_sel=select_bit, src1_sel=select_bit), \
                                       comment="set x to tmp if < 0"))
        elif cDataType.isSingle():
            vgprTemp = self.getVgpr(1)
            module.add(VMulF32(dst=vgpr(Holder(idx=vgprTemp)), src0=sgpr(activationAlpha), src1=self.vgprPrefix(vgprIn), comment="tmp = x * alpha"))
            module.add(VCmpGEF32(dst=VCC(), src0=self.vgprPrefix(vgprIn), src1=0.0, comment="x >= 0 ?"))
            module.add(VCndMaskB32(dst=self.vgprPrefix(vgprOut), src0=vgpr(Holder(idx=vgprTemp)), src1=self.vgprPrefix(vgprIn), comment="set x to tmp if < 0"))
        elif cDataType.isDouble():
            vgprTemp = self.getVgpr(2)
            module.add(VMulF64(dst=vgpr(Holder(idx=vgprTemp), 2), src0=sgpr(activationAlpha, 2), src1=self.vgprPrefix(vgprIn, 2), comment="tmp = x * alpha"))
            module.add(VCmpGEF64(dst=VCC(), src0=self.vgprPrefix(vgprIn, 2), src1=0.0, comment="x >= 0 ?"))
            module.add(VCndMaskB32(dst=self.vgprPrefix(vgprOut), src0=vgpr(Holder(idx=vgprTemp)), src1=self.vgprPrefix(vgprIn), comment="set x to tmp if < 0"))
            module.add(VCndMaskB32(dst=self.vgprPrefix(vgprOut+1), src0=vgpr(Holder(idx=vgprTemp+1)), src1=self.vgprPrefix(vgprIn+1), comment="set x to tmp if < 0"))
        elif cDataType.isInt32():
            vgprTemp = self.getVgpr(1)
            module.add(VMulLOU32(dst=vgpr(Holder(idx=vgprTemp)), src0=sgpr(activationAlpha), src1=self.vgprPrefix(vgprIn), comment="tmp = x * alpha"))
            module.add(VCmpGEI32(dst=VCC(), src0=self.vgprPrefix(vgprIn), src1=0, comment="x >= 0 ?"))
            module.add(VCndMaskB32(dst=self.vgprPrefix(vgprOut), src0=vgpr(Holder(idx=vgprTemp)), src1=self.vgprPrefix(vgprIn), comment="set x to tmp if < 0"))
        else:
            raise RuntimeError("Unsupported data type %s."%cDataType.toDevice("HIP"))
        return module

    def getReluModule(self, cDataType, vgprIn, vgprOut):
        module = Module("LeakyRelu")
        if cDataType.isHalf():
            module.add(VMaxPKF16(dst=self.vgprPrefix(vgprOut), src0=self.vgprPrefix(vgprIn), src1=0, comment="x = max(0, x)" ))
        elif cDataType.isSingle():
            module.add(VMaxF32(dst=self.vgprPrefix(vgprOut), src0=self.vgprPrefix(vgprIn), src1=0, comment="x = max(0, x)" ))
        elif cDataType.isDouble():
            module.add(VMaxF64(dst=self.vgprPrefix(vgprOut, 2), src0=self.vgprPrefix(vgprIn, 2), src1=0, comment="x = max(0, x)" ))
        elif cDataType.isInt32():
            if self.saturateI8:
                vgprTemp = self.getVgpr(1)
                module.add(VMovB32(dst=vgpr(Holder(idx=vgprTemp)), src=hex(127), comment="value = 127"))
                module.add(VMed3I32(dst=self.vgprPrefix(vgprOut), src0=self.vgprPrefix(vgprIn), src1=0, src2=vgpr(Holder(idx=vgprTemp)), comment="x = min(127, max(0, x))" ))
            else:
                module.add(VMaxI32(dst=self.vgprPrefix(vgprOut), src0=self.vgprPrefix(vgprIn), src1=0, comment="x = max(0, x)" ))
        else:
            raise RuntimeError("Unsupported data type %s."%cDataType.toDevice("HIP"))
        return module

    def getSigmoidModule(self, cDataType, vgprIn, vgprOut):
        ti = rocIsa.getInstance()
        self.needCombine = True
        module = Module("Sigmoid")
        if cDataType.isHalf():
            if self.usePK:
                module.add(VMulPKF16(dst=self.vgprPrefix(vgprOut), src0=-1.0, src1=self.vgprPrefix(vgprIn), comment=" x = -x"))
                module.add(self.getExpModule(cDataType, vgprOut, vgprOut))
                module.add(VAddPKF16(dst=self.vgprPrefix(vgprOut), src0=1.0, src1=self.vgprPrefix(vgprOut), \
                                     vop3=VOP3PModifiers(op_sel_hi=[0,1,1]), comment="1 + exp(-x)"))
                for i in range(0, 2):
                    select_bit = SelectBit.WORD_0 if i == 0 else SelectBit.WORD_1
                    module.add(VRcpF16(dst=self.vgprPrefix(vgprOut), src=self.vgprPrefix(vgprOut), \
                                       sdwa=SDWAModifiers(dst_sel=select_bit, dst_unused=UnusedBit.UNUSED_PRESERVE, src0_sel=select_bit), \
                                       comment="1 / (1 + exp(-x))"))
                if ti.getArchCaps()["TransOpWait"]:
                    module.add(SNop(waitState=0, comment="1 wait states"))
            else:
                module.add(VMulF16(dst=self.vgprPrefix(vgprOut), src0=-1.0, src1=self.vgprPrefix(vgprIn), comment=" x = -x"))
                module.add(self.getExpModule(cDataType, vgprOut, vgprOut))
                module.add(VAddF16(dst=self.vgprPrefix(vgprOut), src0=1.0, src1=self.vgprPrefix(vgprOut), comment="1 + exp(-x)"))
                module.add(VRcpF16(dst=self.vgprPrefix(vgprOut), src=self.vgprPrefix(vgprOut), comment="1 / (1 + exp(-x))"))
                if ti.getArchCaps()["TransOpWait"]:
                    module.add(SNop(waitState=0, comment="1 wait states"))
        elif cDataType.isSingle():
            module.add(VMulF32(dst=self.vgprPrefix(vgprOut), src0=-1.0, src1=self.vgprPrefix(vgprIn), comment=" x = -x"))
            module.add(self.getExpModule(cDataType, vgprOut, vgprOut))
            module.add(VAddF32(dst=self.vgprPrefix(vgprOut), src0=1.0, src1=self.vgprPrefix(vgprOut), comment="1 + exp(-x)" ))
            module.add(VRcpF32(dst=self.vgprPrefix(vgprOut), src=self.vgprPrefix(vgprOut), comment="1 / (1 + exp(-x))" ))
            if ti.getArchCaps()["TransOpWait"]:
                module.add(SNop(waitState=0, comment="1 wait states"))
        else:
            raise RuntimeError("Unsupported data type %s."%cDataType.toDevice("HIP"))
        return module

    def getTanhModule(self, cDataType, vgprIn, vgprOut, activationAlpha, activationBeta):
        ti = rocIsa.getInstance()
        self.needCombine = True
        module = Module("Tanh")
        if cDataType.isHalf():
#We don't need s_pack_ll_b32_b16 cause the input is already duplicated
            if self.usePK:
                if activationAlpha:
                    module.add(VMulPKF16(dst=self.vgprPrefix(vgprOut), src0=sgpr(activationAlpha), src1=self.vgprPrefix(vgprIn), comment="x * alpha"))
                    module.add(VMulPKF16(dst=self.vgprPrefix(vgprOut), src0=2, src1=self.vgprPrefix(vgprOut), comment=" x = 2 * x"))
                else:
                    module.add(VMulPKF16(dst=self.vgprPrefix(vgprOut), src0=2, src1=self.vgprPrefix(vgprIn), comment=" x = 2 * x"))
                module.add(self.getExpModule(cDataType, vgprOut, vgprOut))
                module.add(VAddPKF16(dst=self.vgprPrefix(vgprOut), src0=1.0, src1=self.vgprPrefix(vgprOut), \
                                     vop3=VOP3PModifiers(op_sel_hi=[0,1,1]), comment="e^2x + 1"))
                for i in range(0, 2):
                    select_bit = SelectBit.WORD_0 if i == 0 else SelectBit.WORD_1
                    vgprCtrl = "dst_sel:WORD_%d dst_unused:UNUSED_PRESERVE src0_sel:WORD_%d"%(i, i)
                    module.add(VRcpF16(dst=self.vgprPrefix(vgprOut), src=self.vgprPrefix(vgprOut), \
                                       sdwa=SDWAModifiers(dst_sel=select_bit, dst_unused=UnusedBit.UNUSED_PRESERVE, \
                                                          src0_sel=select_bit), \
                                       comment="1 / (1 + exp(-x))"))
                    if ti.getArchCaps()["TransOpWait"]:
                        module.add(SNop(waitState=0, comment="1 wait states")) #workaround for emulator
                module.add(VFmaPKF16(dst=self.vgprPrefix(vgprOut), src0=-2.0, src1=self.vgprPrefix(vgprOut), src2=1.0, \
                                     vop3=VOP3PModifiers(op_sel_hi=[0,1,0,1]), comment="tanh(x) = (1 / (e^2x + 1)) * (-2) + 1"))
                if activationBeta:
                    module.add(VMulPKF16(dst=self.vgprPrefix(vgprOut), src0=sgpr(activationBeta), src1=self.vgprPrefix(vgprOut), comment="beta * tanh(x)"))
            else:
                if activationAlpha:
                    module.add(VMulF16(dst=self.vgprPrefix(vgprOut), src0=sgpr(activationAlpha), src1=self.vgprPrefix(vgprIn), comment="x * alpha"))
                    module.add(VMulF16(dst=self.vgprPrefix(vgprOut), src0=2, src1=self.vgprPrefix(vgprOut), comment=" x = 2 * x"))
                else:
                    module.add(VMulF16(dst=self.vgprPrefix(vgprOut), src0=2, src1=self.vgprPrefix(vgprIn), comment=" x = 2 * x"))
                module.add(self.getExpModule(cDataType, vgprOut, vgprOut))
                module.add(VAddF16(dst=self.vgprPrefix(vgprOut), src0=1.0, src1=self.vgprPrefix(vgprOut), comment="e^2x + 1"))
                module.add(VRcpF16(dst=self.vgprPrefix(vgprOut), src=self.vgprPrefix(vgprOut), comment="1 / (1 + exp(-x))"))
                if ti.getArchCaps()["TransOpWait"]:
                    module.add(SNop(waitState=0, comment="1 wait states")) #workaround for emulator
                module.add(VFmaF16(dst=self.vgprPrefix(vgprOut), src0=-2.0, src1=self.vgprPrefix(vgprOut), src2=1.0, comment="tanh(x) = (1 / (e^2x + 1)) * (-2) + 1"))
                if activationBeta:
                    module.add(VMulF16(dst=self.vgprPrefix(vgprOut), src0=sgpr(activationBeta), src1=self.vgprPrefix(vgprOut), comment="beta * tanh(x)"))
        elif cDataType.isSingle():
            if activationAlpha:
                module.add(VMulF32(dst=self.vgprPrefix(vgprOut), src0=sgpr(activationAlpha), src1=self.vgprPrefix(vgprIn), comment="x * alpha"))
                module.add(VMulF32(dst=self.vgprPrefix(vgprOut), src0=2, src1=self.vgprPrefix(vgprOut), comment=" x = 2 * x"))
            else:
                module.add(VMulF32(dst=self.vgprPrefix(vgprOut), src0=2, src1=self.vgprPrefix(vgprIn), comment=" x = 2 * x"))
            module.add(self.getExpModule(cDataType, vgprOut, vgprOut))
            module.add(VAddF32(dst=self.vgprPrefix(vgprOut), src0=1.0, src1=self.vgprPrefix(vgprOut), comment="e^2x + 1"))
            module.add(VRcpF32(dst=self.vgprPrefix(vgprOut), src=self.vgprPrefix(vgprOut), comment="1 / (e^2x + 1)"))
            if ti.getArchCaps()["TransOpWait"]:
                module.add(SNop(waitState=0, comment="1 wait states")) #workaround for emulator
            module.add(VFmaF32(dst=self.vgprPrefix(vgprOut), src0=-2.0, src1=self.vgprPrefix(vgprOut), src2=1.0, comment="(-2) * (1 / (e^2x + 1)) + 1"))
            if activationBeta:
                module.add(VMulF32(dst=self.vgprPrefix(vgprOut), src0=sgpr(activationBeta), src1=self.vgprPrefix(vgprOut), comment="beta * tanh(x)"))
        else:
            raise RuntimeError("Unsupported data type %s."%cDataType.toDevice("HIP"))
        return module

    def getDGeluModule(self, cDataType, vgprIn, vgprOut):
        ti = rocIsa.getInstance()
        self.needCombine = True
        module = Module("Gradient Gelu")
#x1 = (0.0535161 * pow(x, 3) + 0.398942 * x)
#xx = 0.0356774 * pow(x, 3) + 0.797885 * x
#x2 = cosh - 2(xx) = 4 / pow(math.exp(-xx) + math.exp(xx), 2)
#=> 0.5 * math.tanh(xx) + x1 * x2 + 0.5
        if cDataType.isSingle():
            vgprTemp1 = self.getVgpr(1)
            vgprTemp2 = self.getVgpr(1)
            vgprTemp3 = self.getVgpr(1)
            sgprTemp = self.getSgpr(1)
            module.add(VMulF32(dst=vgpr(Holder(idx=vgprTemp1)), src0=self.vgprPrefix(vgprIn), src1=self.vgprPrefix(vgprIn), comment="tmp1 = pow(x * 2)"))
            module.add(VMulF32(dst=vgpr(Holder(idx=vgprTemp1)), src0=vgpr(Holder(idx=vgprTemp1)), src1=self.vgprPrefix(vgprIn), comment="tmp1 = pow(x * 3)"))
            coef = floatUnion(u=ActivationMagicNumbers["FloatDGeluK1"])
            module.add(VMulF32(dst=vgpr(Holder(idx=vgprTemp2)), src0=hex(coef.u), src1=self.vgprPrefix(vgprIn), comment="tmp2 = 0.398942 * x"))
            coef = floatUnion(u=ActivationMagicNumbers["FloatDGeluK0"])
            module.add(SMovB32(dst=sgpr(Holder(idx=sgprTemp)), src=hex(coef.u), comment="move magic number to sgpr"))
            module.add(VFmaF32(dst=vgpr(Holder(idx=vgprTemp2)), src0=sgpr(Holder(idx=sgprTemp)), src1=vgpr(Holder(idx=vgprTemp1)), src2=vgpr(Holder(idx=vgprTemp2)), comment="tmp2 = 0.0535161 * x^3 + tmp2"))
            coef = floatUnion(u=ActivationMagicNumbers["FloatDGeluK3"])
            module.add(VMulF32(dst=vgpr(Holder(idx=vgprTemp3)), src0=hex(coef.u), src1=self.vgprPrefix(vgprIn), comment="tmp3 = 0.797885 * x"))
            coef = floatUnion(u=ActivationMagicNumbers["FloatDGeluK2"])
            module.add(SMovB32(dst=sgpr(Holder(idx=sgprTemp)), src=hex(coef.u), comment="move magic number to sgpr"))
            module.add(VFmaF32(dst=vgpr(Holder(idx=vgprTemp1)), src0=sgpr(Holder(idx=sgprTemp)), src1=vgpr(Holder(idx=vgprTemp1)), src2=vgpr(Holder(idx=vgprTemp3)), comment="tmp1 = 0.035677 * x^3 + tmp3"))
            module.add(self.getExpModule(cDataType, Holder(idx=vgprTemp1), Holder(idx=vgprTemp3)))
            if self.isAlt:
                module.add(VMulF32(dst=vgpr(Holder(idx=vgprTemp1)), src0=-1.0, src1=vgpr(Holder(idx=vgprTemp1)), comment="tmp1 = -tmp1"))
                module.add(self.getExpModule(cDataType, Holder(idx=vgprTemp1), Holder(idx=vgprTemp1)))
                module.add(VAddF32(dst=self.vgprPrefix(vgprOut), src0=vgpr(Holder(idx=vgprTemp3)), src1=vgpr(Holder(idx=vgprTemp1)), comment="out = e^xx + e^-xx"))
                module.add(VSubF32(dst=vgpr(Holder(idx=vgprTemp1)), src0=vgpr(Holder(idx=vgprTemp3)), src1=vgpr(Holder(idx=vgprTemp1)), comment="tmp1 = e^xx - e^-xx"))
                module.add(VRcpF32(dst=vgpr(Holder(idx=vgprTemp3)), src=self.vgprPrefix(vgprOut), comment="tmp3 = 1/out"))
                if ti.getArchCaps()["TransOpWait"]:
                    module.add(SNop(waitState=0, comment="1 wait states")) #workaround for emulator
                module.add(VMulF32(dst=vgpr(Holder(idx=vgprTemp3)), src0=vgpr(Holder(idx=vgprTemp1)), src1=vgpr(Holder(idx=vgprTemp3)), comment="tmp3 = tmp1 * tmp3"))
                if self.enableGuard:
                    module.add(SMovB32(dst=sgpr(Holder(idx=sgprTemp)), src=hex(0x200), comment="move magic number to sgpr"))
                    module.add(VCmpXClassF32(dst=EXEC(), src0=self.vgprPrefix(vgprOut), src1=sgpr(Holder(idx=sgprTemp)), comment="True if tmp1 = inf"))
                    module.add(VMovB32(dst=vgpr(Holder(idx=vgprTemp3)), src=hex(0x3f800000), comment="tmp3 = 1 if True"))
                    module.add(VCmpXLtF32(dst=EXEC(), src0=vgpr(Holder(idx=vgprTemp2)), src1=0, comment="check if x < 0" ))
                    module.add(VMovB32(dst=vgpr(Holder(idx=vgprTemp3)), src=hex(0xbf800000), comment="tmp3 = -1 if True"))
                    module.add(SSetMask(dst=EXEC(), src=-1, comment="reset mask" ))
                module.add(VMulF32(dst=vgpr(Holder(idx=vgprTemp1)), src0=0.5, src1=vgpr(Holder(idx=vgprTemp3)), comment="tmp1 = 0.5 * tmp1"))
                module.add(VMulF32(dst=self.vgprPrefix(vgprOut), src0=self.vgprPrefix(vgprOut), src1=self.vgprPrefix(vgprOut), comment="out = out * out"))
                module.add(VRcpF32(dst=self.vgprPrefix(vgprOut), src=self.vgprPrefix(vgprOut), comment="out = 1/out"))
                if ti.getArchCaps()["TransOpWait"]:
                    module.add(SNop(waitState=0, comment="1 wait states")) #workaround for emulator
            else:
                module.add(self.getTanhModule(cDataType, Holder(idx=vgprTemp1), vgprOut, "", ""))
                module.add(VMulF32(dst=vgpr(Holder(idx=vgprTemp1)), src0=-1.0, src1=vgpr(Holder(idx=vgprTemp1)), comment="tmp1 = -tmp1"))
                module.add(self.getExpModule(cDataType, Holder(idx=vgprTemp1), Holder(idx=vgprTemp1)))
                module.add(VAddF32(dst=vgpr(Holder(idx=vgprTemp3)), src0=vgpr(Holder(idx=vgprTemp3)), src1=vgpr(Holder(idx=vgprTemp1)), comment="out = e^xx + e^-xx"))
                module.add(VMulF32(dst=vgpr(Holder(idx=vgprTemp1)), src0=0.5, src1=self.vgprPrefix(vgprOut), comment="tmp1 = 0.5 * tmp1"))
                module.add(VMulF32(dst=vgpr(Holder(idx=vgprTemp3)), src0=vgpr(Holder(idx=vgprTemp3)), src1=vgpr(Holder(idx=vgprTemp3)), comment="out = out * out"))
                module.add(VRcpF32(dst=self.vgprPrefix(vgprOut), src=vgpr(Holder(idx=vgprTemp3)), comment="out = 1/out"))
                if ti.getArchCaps()["TransOpWait"]:
                    module.add(SNop(waitState=0, comment="1 wait states")) #workaround for emulator
            coef = floatUnion(f=4)
            module.add(VMulF32(dst=self.vgprPrefix(vgprOut), src0=hex(coef.u), src1=self.vgprPrefix(vgprOut), comment="out = 4 * out"))
            module.add(VFmaF32(dst=self.vgprPrefix(vgprOut), src0=self.vgprPrefix(vgprOut), src1=vgpr(Holder(idx=vgprTemp2)), src2=vgpr(Holder(idx=vgprTemp1)), comment="out = out * tmp2 + tmp1"))
            module.add(VAddF32(dst=self.vgprPrefix(vgprOut), src0=0.5, src1=self.vgprPrefix(vgprOut), comment="out = out + 0.5"))
        else:
            raise RuntimeError("Unsupported data type %s."%cDataType.toDevice("HIP"))
        return module

    def getSiluModule(self, cDataType, vgprIn, vgprOut):
        self.needCombine = True
        module = Module("Silu")
        vgprTemp = self.getVgpr(1)
        module.addModuleAsFlatItems(self.getSigmoidModule(cDataType, vgprIn, Holder(idx=vgprTemp)))
        if cDataType.isHalf():
            if self.usePK:
                mulFunction = VMulPKF16
            else:
                mulFunction = VMulF16
        elif cDataType.isSingle():
            mulFunction = VMulF32
        else:
            raise RuntimeError("Unsupported data type %s."%cDataType.toDevice("HIP"))
        module.add(mulFunction(dst=self.vgprPrefix(vgprOut), src0=self.vgprPrefix(vgprIn), src1=self.vgprPrefix(Holder(idx=vgprTemp)), comment="x / (1 + exp(-x))"))
        return module

    def getSwishModule(self, cDataType, vgprIn, vgprOut, activationAlpha):
        self.needCombine = True
        module = Module("Swish")
        if cDataType.isHalf():
            if self.usePK:
                mulFunction = VMulPKF16
            else:
                mulFunction = VMulF16
        elif cDataType.isSingle():
            mulFunction = VMulF32
        else:
            raise RuntimeError("Unsupported data type %s."%cDataType.toDevice("HIP"))
        vgprTempIn = self.getVgpr(1)
        vgprTempOut = self.getVgpr(1)
        module.add(mulFunction(dst=self.vgprPrefix(Holder(idx=vgprTempIn)), src0=self.vgprPrefix(vgprIn), src1=sgpr(activationAlpha), comment="x * beta"))
        module.addModuleAsFlatItems(self.getSigmoidModule(cDataType, Holder(idx=vgprTempIn), Holder(idx=vgprTempOut)))
        module.add(mulFunction(dst=self.vgprPrefix(vgprOut), src0=self.vgprPrefix(vgprIn), src1=self.vgprPrefix(Holder(idx=vgprTempOut)), comment="x / (1 + exp(-x * beta))"))
        return module

    def getClampModule(self, cDataType, vgprIn, vgprOut, activationAlpha, activationBeta):
        module = Module("Clamp")
        if cDataType.isDouble():
            Vin, Vout = self.vgprPrefix(vgprIn, 2), self.vgprPrefix(vgprOut, 2)
            alpha, beta = sgpr(activationAlpha, 2), sgpr(activationBeta, 2)
        else:
            Vin, Vout = self.vgprPrefix(vgprIn), self.vgprPrefix(vgprOut)
            alpha, beta = sgpr(activationAlpha), sgpr(activationBeta)
        if cDataType.isHalf():
            MIN, MAX = VMinF16, VMaxF16
        elif cDataType.isSingle():
            MIN, MAX = VMinF32, VMaxF32
        elif cDataType.isDouble():
            MIN, MAX = VMinF64, VMaxF64
        elif cDataType.isInt32():
            MIN, MAX = VMinI32, VMaxI32

        if cDataType.isHalf():
            for i in range(0, 2):
                select_bit = SelectBit.WORD_0 if i == 0 else SelectBit.WORD_1
                sdwa = SDWAModifiers(dst_sel = select_bit, dst_unused = UnusedBit.UNUSED_PRESERVE,
                                     src0_sel = select_bit, src1_sel = select_bit)
                module.add(MIN(dst = Vout, src0 = beta, src1 = Vin, sdwa = sdwa, comment = "min(x, beta)"))
                module.add(MAX(dst = Vout, src0 = alpha, src1 = Vout, sdwa = sdwa, comment = "max(alpha, min(x, beta))"))
        else:
            module.add(MIN(dst = Vout, src0 = beta, src1 = Vin, comment = "min(x, beta)"))
            module.add(MAX(dst = Vout, src0 = alpha, src1 = Vout, comment = "max(alpha, min(x, beta))"))
        return module

    ################################################################################
    ################################################################################
    ###
    ###   Cache Functions
    ###
    ################################################################################
    ################################################################################

    def createCache(self, cDataType: DataType, activationType: str, vgprIn, vgprOut, module: Module):
        typeChar = cDataType.toChar()
        if activationType not in self.cacheDict:
            self.cacheDict[activationType] = {}
        actDict = self.cacheDict[activationType]
        copied = deepcopy(module)
#Get reg name
        regName = self.vgprPrefixFormat.split("+")[0] if self.vgprPrefixFormat else ""
        vgprIdxList = createVgprIdxList(copied, [vgprIn, vgprOut], regName)
        actInfo = actCacheInfo(usePK=self.usePK, saturateI8=self.saturateI8, enableGuard=self.enableGuard, \
                isAlt=self.isAlt, prefix=self.vgprPrefixFormat, vgprIdxList=vgprIdxList, module=copied, \
                vgprCounter=self.vgprCounter, sgprCounter=self.sgprCounter)
        if typeChar in actDict:
            actDict[typeChar].append(actInfo)
        else:
            actDict[typeChar] = [actInfo]

    def getCache(self, cDataType: DataType, activationType: str, vgprIn, vgprOut) -> Union[Module, None]:
        if activationType in self.cacheDict:
            actDict = self.cacheDict[activationType]
            typeChar = cDataType.toChar()
            if typeChar in actDict:
                actInfoList = actDict[typeChar]
                for actInfo in actInfoList:
                    if actInfo.isSame(usePK=self.usePK, saturateI8=self.saturateI8, isAlt=self.isAlt, \
                                      enableGuard=self.enableGuard, prefix=self.vgprPrefixFormat):
                        if self.vgprPrefixFormat:
                            for vgpr in actInfo.vgprIdxList[0]:
                                vgpr.regName.setOffset(0, vgprIn)
                            for vgpr in actInfo.vgprIdxList[1]:
                                vgpr.regName.setOffset(0, vgprOut)
                        else:
                            for vgpr in actInfo.vgprIdxList[0]:
                                vgpr.regIdx = vgprIn
                            for vgpr in actInfo.vgprIdxList[1]:
                                vgpr.regIdx = vgprOut
                        self.vgprCounter = actInfo.vgprCounter
                        self.sgprCounter = actInfo.sgprCounter
                        return deepcopy(actInfo.module)
        return None
    } // namespace rocisa
