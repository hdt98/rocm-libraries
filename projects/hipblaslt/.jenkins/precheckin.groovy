@Library('rocJenkins@add_asan') _

import com.amd.project.common.testJob
import org.jenkinsci.plugins.workflow.cps.CpsScript

// Define asan class inline since library branch may not be accessible
class asan extends testJob {
    def asan(CpsScript stageContext) {
        super(stageContext, "hipblaslt", "asan")

        this.project.libraryDependencies = ["hipBLAS-common"]
        this.project.defaults.ccache = false
        this.project.timeout.compile = 600
        this.project.timeout.test = 600

        this.settings.addressSanitizer = true
        this.settings.buildCommand = "./install.sh -c --address-sanitizer"
        this.settings.gtestFilter = "*quick*:*pre_checkin*"
        this.settings.testArgs = ["--test-dir", "/opt/rocm/bin/hipBLASLt"]
        this.settings.testDir = "."
        this.settings.extraCommands["precompile"] = [
            "sudo mkdir -p /home/builder",
            "sudo chown -R jenkins /home/builder"
        ]

        def gpus = stageContext.params?.GPUs?.trim()
        if (gpus) {
            this.settings.buildCommand += " -a ${gpus}"
        }
    }
}

def ci() {
    def job = new asan(this)
    job.ci()
}

return this
