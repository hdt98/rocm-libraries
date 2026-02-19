#!/usr/bin/env groovy
@Library('rocJenkins@pong') _
import com.amd.project.*
import com.amd.docker.*
import java.nio.file.Path;

def runCI =
{
    nodeDetails, jobName->

    def prj = new rocProject('rocRoller', 'Documentation')

    //use docker files from this repo
    prj.repoDockerfile = true
    prj.defaults.ccache = true

    def uniqueTag = params?."Unique Docker image tag" ? org.apache.commons.lang.RandomStringUtils.random(9, true, true) : ""

    def baseParams = rocRollerGetBaseParameters()

    def nodes = new dockerNodes(nodeDetails, jobName, prj)
    nodes.dockerArray.each {
        _, docker ->
        // parameters inherited from target job
        ["ROCROLLER_THEROCK_ROCM_NIGHTLY_URL", "ROCROLLER_THEROCK_ROCM_GFX", "ROCROLLER_THEROCK_ROCM_VERSION"].each {
            param ->
            def value = params?."${param}" ?: baseParams?."${param}";
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

    def commonGroovy

    boolean formatCheck = false

    def buildDocsCommand =
    {
        platform, project->

        commonGroovy = load "${project.paths.project_src_prefix}/.jenkins/common.groovy"
        commonGroovy.runBuildDocsCommand(platform, project)
    }

    buildProject(prj, formatCheck, nodes.dockerArray, buildDocsCommand, null, null)
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

    // URL & GFX family need to be always set, either explicitly or cached,
    // so the latest version can be retrieved. The latest version is used
    // by default unless ROCROLLER_THEROCK_ROCM_VERSION is explicitly set.
    if (!params?.ROCROLLER_THEROCK_ROCM_NIGHTLY_URL?.trim()) {
      error "ROCROLLER_THEROCK_ROCM_NIGHTLY_URL parameter is not set"
    }
    if (!params?.ROCROLLER_THEROCK_ROCM_GFX?.trim()) {
      error "ROCROLLER_THEROCK_ROCM_GFX parameter is not set"
    }
    def therock_rocm_nightly_url = params?.ROCROLLER_THEROCK_ROCM_NIGHTLY_URL
    def therock_rocm_gfx = params?.ROCROLLER_THEROCK_ROCM_GFX
    def latest_therock_rocm_version = params?.ROCROLLER_THEROCK_ROCM_VERSION ?: ["bash", "-c", """
      curl -sL '${therock_rocm_nightly_url}/${therock_rocm_gfx}/rocm/' \
        | grep -oP '<a[^>]*>\\s*rocm-\\K[^<]*(?=\\.tar\\.gz)' \
        | sort -V \
        | tail -1
    """].execute().text.trim()

    def additionalParameters = [
        string(
            name: "ROCROLLER_THEROCK_ROCM_NIGHTLY_URL",
            defaultValue: "${therock_rocm_nightly_url}",
            trim: true,
            description: "URL to retrieve ROCm packages from."
        ),
        string(
            name: "ROCROLLER_THEROCK_ROCM_GFX",
            defaultValue: "${therock_rocm_gfx}",
            trim: true,
            description: "Specify the latest target GFX family for the ROCm packages."
        ),
        string(
            name: "ROCROLLER_THEROCK_ROCM_VERSION",
            defaultValue: "${latest_therock_rocm_version}",
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

    def propertyList = [
        "enterprise":[pipelineTriggers([cron('0 1 * * 0')])],
        "rocm-libraries":[pipelineTriggers([cron('0 1 * * 0')])]
    ]
    propertyList = auxiliary.appendPropertyList(propertyList)

    def jobNameList = [
        "enterprise":([
            "rocroller-ubuntu20-clang":['rocroller-compile', 'rocroller-gfx90a'],
            "rocroller-ubuntu20-gcc":['rocroller-compile', 'rocroller-gfx90a']
        ]),
        "rocm-libraries":([
            "rocroller-ubuntu20-clang":['rocroller-compile', 'rocroller-gfx90a'],
            "rocroller-ubuntu20-gcc":['rocroller-compile', 'rocroller-gfx90a']
        ])
    ]
    jobNameList = auxiliary.appendJobNameList(jobNameList)

    propertyList.each
    {
        jobName, property->
        if (urlJobName == jobName)
            properties(auxiliary.addCommonProperties(property))
    }

    jobNameList.each
    {
        jobName, nodeDetails->
        if (urlJobName == jobName)
            runCI(nodeDetails, jobName)
    }

    if(!jobNameList.keySet().contains(urlJobName))
    {
        properties(auxiliary.addCommonProperties([pipelineTriggers([cron('0 1 * * 6')])]))
        runCI(["rocroller-ubuntu20-clang":['rocroller-compile']], urlJobName)
    }
}
