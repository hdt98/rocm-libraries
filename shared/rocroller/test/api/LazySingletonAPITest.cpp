// Copyright Advanced Micro Devices, Inc., or its affiliates.
// SPDX-License-Identifier:  MIT

#include <filesystem>
#include <string>

#include <catch2/catch_test_macros.hpp>

#include <rocRoller/Expression.hpp> // Needed or else doesn't compile
#include <rocRoller/Operations/Command.hpp>
#include <rocRoller/Utilities/Settings.hpp>

#include <rocRoller/CommandSolution.hpp>

std::shared_ptr<rocRoller::Command> MakeSAXPYCommand()
{
    auto command = std::make_shared<rocRoller::Command>();

    auto dataType = rocRoller::DataType::Float;

    auto xTensorTag = command->addOperation(rocRoller::Operations::Tensor(1, dataType));
    auto xLoadTag   = command->addOperation(rocRoller::Operations::T_Load_Linear(xTensorTag));

    auto yTensorTag = command->addOperation(rocRoller::Operations::Tensor(1, dataType));
    auto yLoadTag   = command->addOperation(rocRoller::Operations::T_Load_Linear(yTensorTag));

    auto alphaScalarTag = command->addOperation(
        rocRoller::Operations::Scalar({dataType, rocRoller::PointerType::PointerGlobal}));
    auto alphaLoadTag = command->addOperation(rocRoller::Operations::T_Load_Scalar(alphaScalarTag));

    auto execute   = rocRoller::Operations::T_Execute(command->getNextTag());
    auto alphaXTag = execute.addXOp(rocRoller::Operations::E_Mul(xLoadTag, alphaLoadTag));
    auto sumTag    = execute.addXOp(rocRoller::Operations::E_Add(alphaXTag, yLoadTag));
    command->addOperation(std::move(execute));

    auto sumTensorTag = command->addOperation(rocRoller::Operations::Tensor(1, dataType));
    command->addOperation(rocRoller::Operations::T_Store_Linear(sumTag, sumTensorTag));

    return command;
}

TEST_CASE("Settings change has observable effect", "[api]")
{
    auto command    = MakeSAXPYCommand();
    auto kernelName = "testKernel";

    SECTION("Generate kernel with default settings")
    {
        auto context = rocRoller::Context::ForTarget(
            rocRoller::GPUArchitectureTarget::fromString("gfx90a"), kernelName);

        auto kernel = rocRoller::CommandKernel(command, kernelName);
        kernel.setContext(context);
        kernel.generateKernel();
    }

    SECTION("Generate kernel with non-default settings that make observable difference")
    {
        std::string tempAsmPath = std::tmpnam(nullptr);
        CHECK(not std::filesystem::exists(tempAsmPath));

        rocRoller::Settings::getInstance()->set(rocRoller::Settings::SaveAssembly, true);
        rocRoller::Settings::getInstance()->set(rocRoller::Settings::AssemblyFile, tempAsmPath);

        auto context = rocRoller::Context::ForTarget(
            rocRoller::GPUArchitectureTarget::fromString("gfx90a"), kernelName);

        auto kernel = rocRoller::CommandKernel(command, kernelName);
        kernel.setContext(context);
        kernel.generateKernel();

        CHECK(std::filesystem::exists(tempAsmPath));
        CHECK(std::filesystem::file_size(tempAsmPath) > 0);
        std::filesystem::remove(tempAsmPath);
    }
}

// Until we have a defined API, please do not add more tests here.
