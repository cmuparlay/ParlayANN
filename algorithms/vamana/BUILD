# Vamana algorithm.

package(
    default_visibility = ["//visibility:public"],
)

cc_library(
    name = "index",
    hdrs = ["index.h"],
    deps = [
        "@parlaylib//parlay:parallel",
        "@parlaylib//parlay:primitives",
        "@parlaylib//parlay:random",
        "//algorithms/utils:beamSearch",
        "//algorithms/utils:graph",
        "//algorithms/utils:point_range",
        "//algorithms/utils:NSGDist",
        "//algorithms/utils:types",
    ],
)

cc_test(
    name = "index_test",
    size = "small",
    srcs = ["index_test.cc"],
    deps = [
        "@googletest//:gtest_main",
        "@parlaylib//parlay:parallel",
        "@parlaylib//parlay:primitives",
        "@parlaylib//parlay:random",
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
        "//algorithms/utils:NSGDist",
        "//algorithms/utils:parse_results",
        "//algorithms/utils:stats",
        "//algorithms/utils:types",
    ],
)

cc_binary(
    name = "vamana_main",
    srcs = ["vamana_main.cc"],
    deps = [
        ":neighbors",
        "@parlaylib//parlay:parallel",
        "@parlaylib//parlay:primitives",
        "@parlaylib//parlay:random",
        "//algorithms/bench:time_loop",
        "//algorithms/bench:parse_command_line",
        "//algorithms/utils:beamSearch",
        "//algorithms/utils:check_nn_recall",
        "//algorithms/utils:NSGDist",
        "//algorithms/utils:euclidian_point",
        "//algorithms/utils:mips_point",
        "//algorithms/utils:graph",
    ],
)

cc_test(
    name = "neighbors_test",
    size = "small",
    srcs = ["neighbors_test.cc"],
    deps = [
        "@googletest//:gtest_main",
        "@parlaylib//parlay:parallel",
        "@parlaylib//parlay:primitives",
        "@parlaylib//parlay:random",
        ":neighbors",
    ],
)
