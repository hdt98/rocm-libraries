// This file is for internal AMD use.
// If you are interested in running your own Jenkins, please raise a github issue for assistance.

def runCompileCommand(platform, project, jobName, boolean sameOrg=false)
{
    project.paths.construct_build_prefix()

    def getDependenciesCommand = ""
    if (project.installLibraryDependenciesFromCI)
    {
        project.libraryDependencies.each
        { libraryName ->
            getDependenciesCommand += auxiliary.getLibrary(libraryName, platform.jenkinsLabel, null, sameOrg)
        }
    }

    String debug = project.buildName.contains('Debug') ? '-g' : ''
    String centos = platform.jenkinsLabel.contains('centos') ? 'source scl_source enable devtoolset-7' : ':'

    def command = """#!/usr/bin/env bash
                set -x
                cd ${project.paths.project_build_prefix}
                ${getDependenciesCommand}
                ${centos}
                LD_LIBRARY_PATH=/opt/rocm/lib ${project.paths.build_command} ${debug}
                """
    platform.runCommand(this, command)
}

def runTestCommand(platform, project, boolean rocmExamples=false)
{
    String buildType = project.buildName.contains('Debug') ? "debug" : "release"
    String testExe = "hipsolver-test"
    def command = """#!/usr/bin/env bash
                    set -x
                    cd ${project.paths.project_build_prefix}/build/${buildType}/clients/staging
                    LD_LIBRARY_PATH=/opt/rocm/lib GTEST_LISTENER=NO_PASS_LINE_IN_LOG ./${testExe} --gtest_output=xml --gtest_color=yes
                """

    platform.runCommand(this, command)
    junit "${project.paths.project_build_prefix}/build/${buildType}/clients/staging/*.xml"

    //ROCM Examples
    if (rocmExamples)
    {
        String buildString = ""
        if (platform.os.contains("ubuntu")){
            buildString += """
                        sudo dpkg -i *.deb
                        sudo apt update
                        sudo apt install -y hipblas-dev
                        """
        }
        else if (platform.os.contains("sles")){
            buildString += """
                        sudo rpm -i *.rpm
                        sudo find /opt -name hipsolver-config.cmake
                        rpm -ql hipsolver-devel
                        sudo zypper refresh || true
                        sudo zypper -n install hipblas-devel
                        """
        }
        else{
            buildString += """
                        sudo rpm -i *.rpm
                        yum list --installed | grep hip
                        sudo find /opt -name hipsolver-config.cmake
                        rpm -ql hipsolver-devel
                        sudo yum -y update
                        sudo yum -y install hipblas-devel
                        """
        }
        String compileCommand = ""
            if (platform.os.contains("rhel")){
                compileCommand = 'cmake -DCMAKE_PREFIX_PATH=/opt/rocm-6.4.0/lib/cmake/hipsolver\;/opt/rocm-6.4.0 -S . -B build'
            }
            else{
                compileCommand = 'cmake -S . -B build'
            }
        testCommand = """#!/usr/bin/env bash
                    set -ex
                    cd ${project.paths.project_build_prefix}/build/release/package
                    ${buildString}
                    cd ../../..
                    testDirs=("Libraries/hipSOLVER")
                    git clone https://github.com/ROCm/rocm-examples.git
                    rocm_examples_dir=\$(readlink -f rocm-examples)
                    for testDir in \${testDirs[@]}; do
                        cd \${rocm_examples_dir}/\${testDir}
                        ${compileCommand}
                        cmake --build build
                        cd ./build
                        ctest --output-on-failure
                    done
                """
        platform.runCommand(this, testCommand, "ROCM Examples")  
    }
}

def runPackageCommand(platform, project, jobName, label='')
{
    def command

    label = label != '' ? '-' + label.toLowerCase() : ''
    String ext = platform.jenkinsLabel.contains('ubuntu') ? "deb" : "rpm"
    String dir = project.buildName.contains('Debug') ? "debug" : "release"

    String testPackageCommand;
    if (platform.jenkinsLabel.contains('ubuntu'))
    {
        testPackageCommand = 'sudo apt-get install -y --simulate '
    }
    else if (platform.jenkinsLabel.contains('centos') || platform.jenkinsLabel.contains('rhel'))
    {
        testPackageCommand = 'sudo yum install -y --setopt tsflags=test '
    }
    else
    {
        testPackageCommand = 'sudo zypper install -y --dry-run --download-only --allow-unsigned-rpm '
    }

    command = """
            set -ex
            cd ${project.paths.project_build_prefix}/build/${dir}
            make package
            ${testPackageCommand} ./hipsolver*.$ext
            mkdir -p package
            if [ ! -z "$label" ]
            then
                for f in hipsolver*.$ext
                do
                    mv "\$f" "hipsolver${label}-\${f#*-}"
                done
            fi
            mv *.${ext} package/
        """

    platform.runCommand(this, command)
    platform.archiveArtifacts(this, """${project.paths.project_build_prefix}/build/${dir}/package/*.${ext}""")
}

return this
