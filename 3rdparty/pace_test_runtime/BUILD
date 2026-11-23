package(default_visibility = ["//visibility:public"])

# Service binaries only: the KVCM SDK remains the branch's existing dependency.
genrule(
    name = "meta_binary",
    testonly = True,
    tags = ["manual"],
    srcs = select({
        "@//:using_cuda12_arm": ["@pace_test_meta_rpm_aarch64//file"],
        "@//:using_arm": ["@pace_test_meta_rpm_aarch64//file"],
        "//conditions:default": ["@pace_test_meta_rpm//file"],
    }),
    outs = ["tair_mempool_meta_service"],
    cmd = """
        stage=$$(mktemp -d)
        trap 'rm -rf "$$stage"' EXIT
        rpm2cpio $(SRCS) | (cd "$$stage" && cpio -idm --quiet)
        cp "$$stage/usr/local/tair_mempool/bin/tair_mempool_meta_service" "$@"
    """,
)

genrule(
    name = "server_binary",
    testonly = True,
    tags = ["manual"],
    srcs = select({
        "@//:using_cuda12_arm": ["@pace_test_server_rpm_aarch64//file"],
        "@//:using_arm": ["@pace_test_server_rpm_aarch64//file"],
        "//conditions:default": ["@pace_test_server_rpm//file"],
    }),
    outs = ["tair_mempool_server", "libtent_shared.so.5"],
    cmd = """
        stage=$$(mktemp -d)
        trap 'rm -rf "$$stage"' EXIT
        rpm2cpio $(SRCS) | (cd "$$stage" && cpio -idm --quiet)
        cp "$$stage/usr/local/tair_mempool/bin/tair_mempool_server" "$(@D)/tair_mempool_server"
        cp -L "$$stage/usr/local/tair_mempool/lib/libtent_shared.so.5" "$(@D)/libtent_shared.so.5"
    """,
)

filegroup(
    name = "runtime",
    testonly = True,
    tags = ["manual"],
    srcs = [":meta_binary", ":server_binary", ":consumer_binary"],
)

genrule(
    name = "consumer_binary",
    testonly = True,
    tags = ["manual"],
    srcs = select({
        "@//:using_cuda12_arm": ["@pace_test_consumer_rpm_aarch64//file"],
        "@//:using_arm": ["@pace_test_consumer_rpm_aarch64//file"],
        "//conditions:default": ["@pace_test_consumer_rpm//file"],
    }),
    outs = ["consumer/tair_mempool_server", "consumer/libtent_shared.so.5"],
    cmd = """
        stage=$$(mktemp -d)
        trap 'rm -rf "$$stage"' EXIT
        rpm2cpio $(SRCS) | (cd "$$stage" && cpio -idm --quiet)
        mkdir -p "$(@D)/consumer"
        cp "$$stage/usr/local/tair_mempool/bin/tair_mempool_server" "$(@D)/consumer/tair_mempool_server"
        cp -L "$$stage/usr/local/tair_mempool/lib/libtent_shared.so.5" "$(@D)/consumer/libtent_shared.so.5"
    """,
)
