// This file is for internal AMD use.
// If you are interested in running your own Jenkins, please raise a github issue for assistance.

def rocRollerGetRunParams(baseParams, jobParams)
{
    // URL & GFX family need to be always set, either explicitly or cached,
    // so the latest version can be retrieved. The latest version is used
    // by default unless ROCROLLER_THEROCK_ROCM_VERSION is explicitly set.
    if (!jobParams?.ROCROLLER_THEROCK_ROCM_NIGHTLY_URL?.trim() && !baseParams?.ROCROLLER_THEROCK_ROCM_NIGHTLY_URL?.trim()) {
      error "ROCROLLER_THEROCK_ROCM_NIGHTLY_URL parameter is not set"
    }
    if (!jobParams?.ROCROLLER_THEROCK_ROCM_GFX?.trim() && !baseParams?.ROCROLLER_THEROCK_ROCM_GFX?.trim()) {
      error "ROCROLLER_THEROCK_ROCM_GFX parameter is not set"
    }
    def therock_rocm_nightly_url = jobParams?.ROCROLLER_THEROCK_ROCM_NIGHTLY_URL ?: baseParams?.ROCROLLER_THEROCK_ROCM_NIGHTLY_URL
    def therock_rocm_gfx = jobParams?.ROCROLLER_THEROCK_ROCM_GFX ?: baseParams?.ROCROLLER_THEROCK_ROCM_GFX
    def latest_therock_rocm_version = jobParams?.ROCROLLER_THEROCK_ROCM_VERSION ?: baseParams?.ROCROLLER_THEROCK_ROCM_VERSION

    latest_therock_rocm_version = latest_therock_rocm_version ?: ["bash", "-c", """
      curl -sL '${therock_rocm_nightly_url}/${therock_rocm_gfx}/rocm/' \
        | grep -oP '<a[^>]*>\\s*rocm-\\K[^<]*(?=\\.tar\\.gz)' \
        | sort -V \
        | tail -1
    """].execute().text.trim()

    return [
        "ROCROLLER_THEROCK_ROCM_NIGHTLY_URL": therock_rocm_nightly_url,
        "ROCROLLER_THEROCK_ROCM_GFX": therock_rocm_gfx,
        "ROCROLLER_THEROCK_ROCM_VERSION": latest_therock_rocm_version
    ];
}

return this
