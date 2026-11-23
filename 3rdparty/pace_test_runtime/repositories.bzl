load("@bazel_tools//tools/build_defs/repo:http.bzl", "http_archive", "http_file")

def pace_test_deps():
    http_file(
        name = "pace_test_meta_rpm",
        urls = [
            "https://ais-zhouyi-oss1.oss-cn-shenzhen.aliyuncs.com/PACE/release/tair_mempool_metaservice-3.1.21_2026_11_18_16_18-1.al8.x86_64.rpm",
        ],
        sha256 = "4556d4827743178d1799befabc8dc8c917f376dd59ed453085c07377686f20d3",
    )

    http_file(
        name = "pace_test_meta_rpm_aarch64",
        urls = [
            "https://ais-zhouyi-oss1.oss-cn-shenzhen.aliyuncs.com/PACE/release/tair_mempool_metaservice-3.1.21_2026_11_18_16_21-1.al8.aarch64.rpm",
        ],
        sha256 = "84225a39540dfd5cb55d144c4cf6722d213b570dc9d2c0c57fcf8aa64a8fc8f3",
    )

    http_file(
        name = "pace_test_server_rpm",
        urls = [
            "https://ais-zhouyi-oss1.oss-cn-shenzhen.aliyuncs.com/PACE/release/tair_mempool_cssdv_server-3.1.21_2026_11_18_16_24_qp_affinity_mlnx58.9-1.al8.x86_64.rpm",
        ],
        sha256 = "eb2f9657aca1dc61a08dab77e2fac9e16c47885f804d92cf0982332f74a8e7a3",
    )

    http_file(
        name = "pace_test_server_rpm_aarch64",
        urls = [
            "https://ais-zhouyi-oss1.oss-cn-shenzhen.aliyuncs.com/PACE/release/tair_mempool_cssdv_server-3.1.21_2026_11_18_16_26_qp_affinity_mlnx58.9-1.al8.aarch64.rpm",
        ],
        sha256 = "0db40df17c0ff2ed6de8408ac673b9662c69f0f8f6a83023f2ed103ff4e5eced",
    )

    http_file(
        name = "pace_test_consumer_rpm",
        urls = [
            "https://artlab.alibaba-inc.com/1/GENERIC/pace-releases/files/PACE/release/tair_mempool_server-3.1.21_2026_11_20_10_40_50_f14290c5f767_71a5d25e_qp_affinity_mlnx58.9-1.al8.x86_64.rpm?version=3.1.21",
        ],
        sha256 = "b0c111eaff1404205dd4cdd9dcba95085f11919e9cf14dc759c80ea2e09ddc2e",
    )

    http_file(
        name = "pace_test_consumer_rpm_aarch64",
        urls = [
            "https://artlab.alibaba-inc.com/1/GENERIC/pace-releases/files/PACE/release/tair_mempool_server-3.1.21_2026_11_20_10_44_00_f14290c5f767_f94fe727_qp_affinity_mlnx58.9-1.al8.aarch64.rpm?version=3.1.21",
        ],
        sha256 = "d465060856640cf86f5ee751bfc7b524745aa6990dad2f3b4909ea3635a35286",
    )

    http_file(
        name = "pace_test_kvcm_sdk_rpm_cuda130_x86",
        urls = [
            "https://rtp-maga.oss-cn-zhangjiakou.aliyuncs.com/kv_cache_manager/client/release/qwen4_dpsk_1116/1dcf928b-ea8b7d8a/kv-cache-manager-client-2026_11_20_16_05-cuda130.x86_64.rpm",
        ],
        sha256 = "93308ad49823ab1e94f2755cbbf6de530c9f2a5e739bcccb8982aff8dffba125",
    )

    http_file(
        name = "pace_test_kvcm_sdk_rpm_cuda130_arm",
        urls = [
            "https://rtp-maga.oss-cn-zhangjiakou.aliyuncs.com/kv_cache_manager/client/release/qwen4_dpsk_1116/1dcf928b-ea8b7d8a/kv-cache-manager-client-2026_11_20_16_05-cuda130.aarch64.rpm",
        ],
        sha256 = "97cd811df8bd6b95d9edcc2013131b077d335a8bb35b6db6d9e46f3de5a024b0",
    )

    for architecture, digest in [
        ("x86", "a8332a4e6e44110cd4d299071c3c5320a90f736b141454733777f900062c6277"),
        ("arm", "a309f856fe19744e9781027e02e704ea49d1092cfc96d2b282b3843c2400876d"),
    ]:
        http_archive(
            name = "pace_test_kvcm_manager_" + architecture,
            urls = [
                "https://hub.docker.alibaba-inc.com/v2/isearch/kv_cache_manager_prod_dashllm/blobs/sha256:" + digest,
            ],
            sha256 = digest,
            type = "tar",
            strip_prefix = "home/admin/kv_cache_manager",
            patch_cmds = ["tar -xzf kv_cache_manager_server.tar.gz"],
            build_file_content = 'exports_files(["bin/kv_cache_manager_bin"])',
        )
