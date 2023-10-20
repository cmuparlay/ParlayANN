load("@bazel_tools//tools/build_defs/repo:http.bzl", "http_archive")
load("@bazel_tools//tools/cpp:cc_configure.bzl", "cc_configure")

cc_configure()

http_archive(
    name = "parlaylib",
    sha256 = "68c062ad116fd49d77651d7a24fb985aa66e8ec9ad05176b6af3ab5d29a16b1f",
    strip_prefix = "parlaylib-bazel/include/",
    urls = ["https://github.com/ParAlg/parlaylib/archive/refs/tags/bazel.tar.gz"],
)

http_archive(
    name = "googletest",
    sha256 = "b4870bf121ff7795ba20d20bcdd8627b8e088f2d1dab299a031c1034eddc93d5",
    strip_prefix = "googletest-release-1.11.0",
    urls = ["https://github.com/google/googletest/archive/release-1.11.0.tar.gz"],
)

http_archive(
    name = "benchmark",
    urls = ["https://github.com/google/benchmark/archive/v1.8.3.tar.gz"],
    strip_prefix = "benchmark-1.8.3",
)