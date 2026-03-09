@Library('rocJenkins@add_asan') _

def ci() {
    def job = new com.amd.project.hipblaslt.asan(this)
    job.ci()
}

return this
