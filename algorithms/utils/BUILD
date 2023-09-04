# ANNS utilility.

package(default_visibility = ["//algorithms:__subpackages__"])

cc_library(
    name = "types",
    hdrs = ["types.h"],
    deps = [
        "@parlaylib//parlay:parallel",
        "@parlaylib//parlay:primitives",
    ],
)

cc_library(
    name = "NSGDist",
    hdrs = ["NSGDist.h"],
    deps = [
        "@parlaylib//parlay:parallel",
        "@parlaylib//parlay:primitives",
    ],
)

cc_library(
    name = "beamSearch",
    hdrs = ["beamSearch.h"],
    deps = [
        "@parlaylib//parlay:io",
        "@parlaylib//parlay:parallel",
        "@parlaylib//parlay:primitives",
        "@parlaylib//parlay:random",
        ":indexTools",
        ":types",
        ":NSGDist",
    ],
)

cc_library(
    name = "stats",
    hdrs = ["stats.h"],
    deps = [
        "@parlaylib//parlay:parallel",
        "@parlaylib//parlay:primitives",
        ":indexTools",
        ":types",
    ],
)

cc_library(
    name = "parse_results",
    hdrs = ["parse_results.h"],
    deps = [
        "@parlaylib//parlay:parallel",
        "@parlaylib//parlay:primitives",
        ":indexTools",
    ],
)

cc_library(
    name = "check_nn_recall",
    hdrs = ["check_nn_recall.h"],
    deps = [
        "@parlaylib//parlay:parallel",
        "@parlaylib//parlay:primitives",
        ":beamSearch",
        ":csvfile",
        ":indexTools",
        ":parse_results",
        ":types",
    ],
)

cc_library(
    name = "csvfile",
    hdrs = ["csvfile.h"],
)
