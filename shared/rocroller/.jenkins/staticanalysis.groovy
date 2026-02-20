#!/usr/bin/env groovy
// This shared library is available at https://github.com/ROCmSoftwarePlatform/rocJENKINS/
@Library('rocJenkins@pong') _

// This is file for internal AMD use.
// If you are interested in running your own Jenkins, please raise a github issue for assistance.

import com.amd.project.*
import com.amd.docker.*
import java.nio.file.Path


def runCI =
{
    nodeDetails, jobName ->

    def prj = new rocProject('rocRoller', 'StaticAnalysis')

    def uniqueTag = params?."Unique Docker image tag" ? org.apache.commons.lang.RandomStringUtils.random(9, true, true) : ""

    def helpersGroovy
    def runParams
    node {
        checkout scm
        helpersGroovy = load "${WORKSPACE}/shared/rocroller/.jenkins/helpers.groovy"
    }
    runParams = helpersGroovy.rocRollerGetRunParams(rocRollerGetBaseParameters(), params)

    def nodes = new dockerNodes(nodeDetails, jobName, prj)
    nodes.dockerArray.each {
        _, docker ->
        // parameters inherited from target job
        ["ROCROLLER_THEROCK_ROCM_NIGHTLY_URL", "ROCROLLER_THEROCK_ROCM_GFX", "ROCROLLER_THEROCK_ROCM_VERSION"].each {
            param ->
            def value = runParams."${param}";
            if (value)
            {
                docker.buildArgs += " --build-arg ${param}=${value}"
            }
        }

        if (uniqueTag)
        {
            docker.customFinalTag = uniqueTag
        }
    }

    boolean formatCheck = true
    boolean staticAnalysis = false

    def commonGroovy

    def compileCommand =
    {
        platform, project->
        commonGroovy = load "${project.paths.project_src_prefix}/.jenkins/common.groovy"
        commonGroovy.runCompileCommand(platform, project, jobName, false, false, '', true, true)
    }

    // change first null to compileCommand once pytest-cmake is available
    buildProject(prj, formatCheck, nodes.dockerArray, null, null, null, staticAnalysis)
}

def rocRollerGetBaseParameters() {
    def baseParameters = jenkins.model.Jenkins.instance.getItemByFullName(env.JOB_NAME)
        .parent.getJob(env.CHANGE_TARGET)
        ?.getProperty(hudson.model.ParametersDefinitionProperty)
        ?.parameterDefinitions
        ?.collect {[ it.name, it.defaultParameterValue.value]}
        ?.collectEntries();
    return baseParameters;
}

ci: {
    String urlJobName = auxiliary.getTopJobName(env.BUILD_URL)

    def additionalParameters = [
        string(
            name: "ROCROLLER_THEROCK_ROCM_NIGHTLY_URL",
            defaultValue: params?.ROCROLLER_THEROCK_ROCM_NIGHTLY_URL,
            trim: true,
            description: "URL to retrieve ROCm packages from."
        ),
        string(
            name: "ROCROLLER_THEROCK_ROCM_GFX",
            defaultValue: params?.ROCROLLER_THEROCK_ROCM_GFX,
            trim: true,
            description: "Specify the latest target GFX family for the ROCm packages."
        ),
        string(
            name: "ROCROLLER_THEROCK_ROCM_VERSION",
            defaultValue: params?.ROCROLLER_THEROCK_ROCM_VERSION,
            trim: true,
            description: "Specify the ROCm version to use."
        ),
        booleanParam(
            name: "Unique Docker image tag",
            defaultValue: false,
            description: "Whether to tag the built docker image with a unique tag. WARNING: Use sparingly, each unique tag costs significant storage space."
        )
    ]
    auxiliary.registerAdditionalParameters(additionalParameters)

    properties(auxiliary.addCommonProperties([pipelineTriggers([cron('0 12 * * 6')])]))

    def jobNameList = [
        "enterprise":(["ubuntu20":['rocroller-compile']]),
        "rocm-libraries":(["ubuntu20":['rocroller-compile']])
    ]
    jobNameList = auxiliary.appendJobNameList(jobNameList)

    jobNameList.each
    {
        jobName, nodeDetails->
        if (urlJobName == jobName)
            runCI(nodeDetails, jobName)
    }
}
