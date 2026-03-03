// Copyright © Advanced Micro Devices, Inc., or its affiliates.
// SPDX-License-Identifier:  MIT

#include "gtest_common.hpp"

using namespace sizer;

namespace {
struct Size
{
    Size(size_t _size) : size(_size) {}
    Size(size_t _mant, size_t _exp) : size(_mant * (1ULL << _exp)) {}
    size_t size;

    template <typename T>
    static std::vector<Size> GetElements()
    {
        // clang-format off
        return {
            { 0ULL },
            { 2ULL },
            { 4ULL },
            { 1, 32 },
            { 4, 32 }
        };
        // clang-format on
    }
};
} // namespace

template <typename T>
struct SizerBase
{
    SizerBase(size_t _bytes) : elements(_bytes / sizeof(T)), bytes(_bytes) {}
    virtual ~SizerBase() {}

    const size_t elements;
    const size_t bytes;
    virtual const bool CheckEobElements(size_t _elements) const   = 0;
    virtual const bool CheckEobBytes(size_t _elements) const      = 0;
    virtual const bool CheckAboveElements(size_t _elements) const = 0;
    virtual const bool CheckAboveBytes(size_t _elements) const    = 0;
};

template <typename T, size_t MAX_BYTES>
struct Sizers : public SizerBase<T>
{
    Sizers() : SizerBase<T>(MAX_BYTES)
    {
        eob = std::make_unique<EqualOrBelow<T, MAX_BYTES>>();
        abv = std::make_unique<Above<T, MAX_BYTES>>();
    }
    virtual ~Sizers() {}

    std::unique_ptr<EqualOrBelow<T, MAX_BYTES>> eob;
    std::unique_ptr<Above<T, MAX_BYTES>> abv;

    virtual const bool CheckEobElements(size_t _elements) const
    {
        return eob->CheckElements(_elements);
    }
    virtual const bool CheckEobBytes(size_t _bytes) const { return eob->CheckBytes(_bytes); }
    virtual const bool CheckAboveElements(size_t _elements) const
    {
        return abv->CheckElements(_elements);
    }
    virtual const bool CheckAboveBytes(size_t _bytes) const { return abv->CheckBytes(_bytes); }
};

template <typename T>
struct ConfigSizer : public ::testing::TestWithParam<Size>
{
protected:
    void RunTests()
    {
        size_t elements = this->GetParam().size;
        size_t bytes    = elements * sizeof(T);

        std::vector<std::unique_ptr<SizerBase<T>>> sizers;
        sizers.push_back(std::make_unique<Sizers<T, 1ULL * sizeof(T)>>());
        sizers.push_back(std::make_unique<Sizers<T, 2ULL * sizeof(T)>>());
        sizers.push_back(std::make_unique<Sizers<T, 3ULL * sizeof(T)>>());
        sizers.push_back(std::make_unique<Sizers<T, 5ULL * sizeof(T)>>());
        sizers.push_back(std::make_unique<Sizers<T, ((1ULL << 32) - 1) * sizeof(T)>>());
        sizers.push_back(std::make_unique<Sizers<T, (1ULL << 32) * sizeof(T)>>());
        sizers.push_back(std::make_unique<Sizers<T, ((1ULL << 32) + 1) * sizeof(T)>>());
        sizers.push_back(std::make_unique<Sizers<T, ((1ULL << 34) - 1) * sizeof(T)>>());
        sizers.push_back(std::make_unique<Sizers<T, (1ULL << 34) * sizeof(T)>>());
        sizers.push_back(std::make_unique<Sizers<T, ((1ULL << 34) + 1) * sizeof(T)>>());

        for(const auto& szr : sizers)
        {
            EXPECT_EQUAL(elements <= szr->elements, szr->CheckEobElements(elements));
            EXPECT_EQUAL(bytes <= szr->bytes, szr->CheckEobBytes(bytes));
            EXPECT_EQUAL(elements > szr->elements, szr->CheckAboveElements(elements));
            EXPECT_EQUAL(bytes > szr->bytes, szr->CheckAboveBytes(bytes));
        }
    }
};

// The exact types tested do not matter; only their size.
using CPU_TestConfigSizer_I8   = ConfigSizer<int8_t>;
using CPU_TestConfigSizer_I16  = ConfigSizer<int16_t>;
using CPU_TestConfigSizer_I32  = ConfigSizer<int32_t>;
using CPU_TestConfigSizer_FP64 = ConfigSizer<double>;

TEST_P(CPU_TestConfigSizer_I8, Tests) { RunTests(); }
TEST_P(CPU_TestConfigSizer_I16, Tests) { RunTests(); }
TEST_P(CPU_TestConfigSizer_I32, Tests) { RunTests(); }
TEST_P(CPU_TestConfigSizer_FP64, Tests) { RunTests(); }

INSTANTIATE_TEST_SUITE_P(Smoke,
                         CPU_TestConfigSizer_I8,
                         testing::ValuesIn(Size::GetElements<int8_t>()));
INSTANTIATE_TEST_SUITE_P(Smoke,
                         CPU_TestConfigSizer_I16,
                         testing::ValuesIn(Size::GetElements<int16_t>()));
INSTANTIATE_TEST_SUITE_P(Smoke,
                         CPU_TestConfigSizer_I32,
                         testing::ValuesIn(Size::GetElements<int32_t>()));
INSTANTIATE_TEST_SUITE_P(Smoke,
                         CPU_TestConfigSizer_FP64,
                         testing::ValuesIn(Size::GetElements<double>()));
