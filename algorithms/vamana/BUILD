# Vamana algorithm.

cc_library(
    name = "index",
    hdrs = ["index.h"],
    deps = [
        "@parlaylib//parlay:parallel",
        "@parlaylib//parlay:primitives",
        "@parlaylib//parlay:random",
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

cc_library(
    name = "neighbors",
    hdrs = ["neighbors.h"],
    deps = [
        ":index",
        "@parlaylib//parlay:parallel",
        "@parlaylib//parlay:primitives",
        "@parlaylib//parlay:random",
        "//algorithms/utils:beamSearch",
        "//algorithms/utils:check_nn_recall",
        "//algorithms/utils:csvfile",
        "//algorithms/utils:NSGDist",
        "//algorithms/utils:parse_results",
        "//algorithms/utils:stats",
        "//algorithms/utils:types",
    ],
)

cc_test(
    name = "neighbors_test",
    size = "small",
    srcs = ["neighbors_test.cc"],
    deps = [
        "@googletest//:gtest_main",
        ":neighbors",
    ],
)
