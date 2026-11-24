load("@bazel_tools//tools/build_defs/repo:http.bzl", "http_archive", "http_file")

PACE_TEST_ARTIFACTS = {
    "rpms": {
        "pace_test_meta_rpm": (
            "https://ais-zhouyi-oss1.oss-cn-shenzhen.aliyuncs.com/PACE/release/tair_mempool_metaservice-3.1.21_2026_11_18_16_18-1.al8.x86_64.rpm",
            "4556d4827743178d1799befabc8dc8c917f376dd59ed453085c07377686f20d3",
        ),
        "pace_test_meta_rpm_aarch64": (
            "https://ais-zhouyi-oss1.oss-cn-shenzhen.aliyuncs.com/PACE/release/tair_mempool_metaservice-3.1.21_2026_11_18_16_21-1.al8.aarch64.rpm",
            "84225a39540dfd5cb55d144c4cf6722d213b570dc9d2c0c57fcf8aa64a8fc8f3",
        ),
        "pace_test_server_rpm": (
            "https://ais-zhouyi-oss1.oss-cn-shenzhen.aliyuncs.com/PACE/release/tair_mempool_cssdv_server-3.1.21_2026_11_18_16_24_qp_affinity_mlnx58.9-1.al8.x86_64.rpm",
            "eb2f9657aca1dc61a08dab77e2fac9e16c47885f804d92cf0982332f74a8e7a3",
        ),
        "pace_test_server_rpm_aarch64": (
            "https://ais-zhouyi-oss1.oss-cn-shenzhen.aliyuncs.com/PACE/release/tair_mempool_cssdv_server-3.1.21_2026_11_18_16_26_qp_affinity_mlnx58.9-1.al8.aarch64.rpm",
            "0db40df17c0ff2ed6de8408ac673b9662c69f0f8f6a83023f2ed103ff4e5eced",
        ),
        "pace_test_consumer_rpm": (
            "https://artlab.alibaba-inc.com/1/GENERIC/pace-releases/files/PACE/release/tair_mempool_server-3.1.21_2026_11_20_10_40_50_f14290c5f767_71a5d25e_qp_affinity_mlnx58.9-1.al8.x86_64.rpm?version=3.1.21",
            "b0c111eaff1404205dd4cdd9dcba95085f11919e9cf14dc759c80ea2e09ddc2e",
        ),
        "pace_test_consumer_rpm_aarch64": (
            "https://artlab.alibaba-inc.com/1/GENERIC/pace-releases/files/PACE/release/tair_mempool_server-3.1.21_2026_11_20_10_44_00_f14290c5f767_f94fe727_qp_affinity_mlnx58.9-1.al8.aarch64.rpm?version=3.1.21",
            "d465060856640cf86f5ee751bfc7b524745aa6990dad2f3b4909ea3635a35286",
        ),
        "pace_test_kvcm_sdk_rpm_cuda130_x86": (
            "https://rtp-maga.oss-cn-zhangjiakou.aliyuncs.com/kv_cache_manager/client/release/qwen4_dpsk_1116/1dcf928b-ea8b7d8a/kv-cache-manager-client-2026_11_20_16_05-cuda130.x86_64.rpm",
            "93308ad49823ab1e94f2755cbbf6de530c9f2a5e739bcccb8982aff8dffba125",
        ),
        "pace_test_kvcm_sdk_rpm_cuda130_arm": (
            "https://rtp-maga.oss-cn-zhangjiakou.aliyuncs.com/kv_cache_manager/client/release/qwen4_dpsk_1116/1dcf928b-ea8b7d8a/kv-cache-manager-client-2026_11_20_16_05-cuda130.aarch64.rpm",
            "97cd811df8bd6b95d9edcc2013131b077d335a8bb35b6db6d9e46f3de5a024b0",
        ),
    },
    "rpm_layers": {
        "pace_test_meta_rpm": (
            "hub.docker.alibaba-inc.com/system-analysis/tair-mempool-metaservice@sha256:a0ab4b9a4ee2bfe3b879c0b652c2fdddfe8eb55b67ecfeb5bddb7e616e58f40a",
            "tair_mempool/tair_mempool_metaservice-3.1.21-1.al8.x86_64.rpm",
        ),
        "pace_test_server_rpm": (
            "hub.docker.alibaba-inc.com/system-analysis/tair-mempool-server-qp-affinity@sha256:4f98c78baa5afcc74d7e62666476eb59acda6eb5cf20fa8f524d255618c5c204",
            "tair_mempool/tair_mempool_cssdv_server-3.1.21-1.al8.x86_64.rpm",
        ),
    },
    "manager_repo": "hub.docker.alibaba-inc.com/isearch/kv_cache_manager_prod_dashllm",
    "manager_layers": {
        "x86": "a8332a4e6e44110cd4d299071c3c5320a90f736b141454733777f900062c6277",
        "arm": "a309f856fe19744e9781027e02e704ea49d1092cfc96d2b282b3843c2400876d",
    },
}

def pace_test_deps():
    for name, (url, digest) in PACE_TEST_ARTIFACTS["rpms"].items():
        http_file(name = name, urls = [url], sha256 = digest)

    registry, image = PACE_TEST_ARTIFACTS["manager_repo"].split("/", 1)
    for architecture, digest in PACE_TEST_ARTIFACTS["manager_layers"].items():
        http_archive(
            name = "pace_test_kvcm_manager_" + architecture,
            urls = [
                "https://%s/v2/%s/blobs/sha256:%s" % (registry, image, digest),
            ],
            sha256 = digest,
            type = "tar",
            strip_prefix = "home/admin/kv_cache_manager",
            patch_cmds = ["tar -xzf kv_cache_manager_server.tar.gz"],
            build_file_content = 'exports_files(["bin/kv_cache_manager_bin"])',
        )
