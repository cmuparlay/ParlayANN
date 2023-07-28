# Vamana algorithm.

cc_library(
    name = "index",
    hdrs = ["index.h"],
    deps = [
        "@parlaylib//parlay:parallel",
        "//algorithms/utils:indexTools",
        "//algorithms/utils:NSGDist",
    ],
)

cc_test(
    name = "index_test",
    size = "small",
    srcs = ["index_test.cc"],
    deps = [
        "@googletest//:gtest_main",
        ":index",
    ],
)

