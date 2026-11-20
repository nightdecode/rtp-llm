load("@bazel_tools//tools/build_defs/repo:http.bzl", "http_file")

def pace_test_deps():
    http_file(
        name = "pace_test_meta_rpm",
        urls = [
            "https://rtp-maga.cn-zhangjiakou.oss.aliyuncs.com/pace/server/tair_mempool_metaservice-3.1.21_2026_10_13_15_39-1.al8.x86_64.rpm",
            "https://rtp-maga.oss-cn-zhangjiakou.aliyuncs.com/pace/server/tair_mempool_metaservice-3.1.21_2026_10_13_15_39-1.al8.x86_64.rpm",
        ],
        sha256 = "5754d2ede183b2173d963f15d3561b710eb5e9ce52a4b081875bf9e80ba20133",
    )

    http_file(
        name = "pace_test_meta_rpm_aarch64",
        urls = [
            "https://rtp-maga.cn-zhangjiakou.oss.aliyuncs.com/pace/server/tair_mempool_metaservice-3.1.21_2026_10_13_15_45-1.al8.aarch64.rpm",
            "https://rtp-maga.oss-cn-zhangjiakou.aliyuncs.com/pace/server/tair_mempool_metaservice-3.1.21_2026_10_13_15_45-1.al8.aarch64.rpm",
        ],
        sha256 = "5944ef9f7c89bfc0f3e04afde8c0f536387dfc60d897d460d699a0e792336efd",
    )

    http_file(
        name = "pace_test_server_rpm",
        urls = [
            "https://rtp-maga.cn-zhangjiakou.oss.aliyuncs.com/pace/server/ci-83025525/tair_mempool_cssdv_server-3.1.21-1.al8.x86_64.rpm",
            "https://rtp-maga.oss-cn-zhangjiakou.aliyuncs.com/pace/server/ci-83025525/tair_mempool_cssdv_server-3.1.21-1.al8.x86_64.rpm",
        ],
        sha256 = "e7cb9a84843d7f71c0e3281493ddf4ea99e0c6ddbba92e553baf6e6712913778",
    )

    http_file(
        name = "pace_test_server_rpm_aarch64",
        urls = [
            "https://rtp-maga.cn-zhangjiakou.oss.aliyuncs.com/pace/server/ci-83025525/tair_mempool_cssdv_server-3.1.21-1.al8.aarch64.rpm",
            "https://rtp-maga.oss-cn-zhangjiakou.aliyuncs.com/pace/server/ci-83025525/tair_mempool_cssdv_server-3.1.21-1.al8.aarch64.rpm",
        ],
        sha256 = "5367ee25d0c27dff69e107489d40803ae4a2fca41762283525dce80b551e4215",
    )
