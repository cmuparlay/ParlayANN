# ANNS utilility.

package(default_visibility = ["//algorithms:__subpackages__"])


cc_library(
    name = "csvfile",
    hdrs = ["csvfile.h"],
)

cc_library(
    name = "beamSearch",
    hdrs = ["beamSearch.h"],
    deps = [
        "@parlaylib//parlay:io",
        "@parlaylib//parlay:parallel",
        "@parlaylib//parlay:primitives",
        "@parlaylib//parlay:random",
        ":types",
        ":graph",
        ":stats",
    ],
)

cc_library(
    name = "check_range_recall",
    hdrs = ["check_nn_recall.h"],
    deps = [
        "@parlaylib//parlay:parallel",
        "@parlaylib//parlay:primitives",
        ":beamSearch",
        ":csvfile",
        ":parse_results",
        ":stats",
        ":types",
    ],
)

# cc_library(
#     name = "beamSearch",
#     hdrs = ["beamSearch.h"],
#     deps = [
#         "@parlaylib//parlay:io",
#         "@parlaylib//parlay:parallel",
#         "@parlaylib//parlay:primitives",
#         "@parlaylib//parlay:random",
#         ":indexTools",
#         ":types",
#     ],
# )

# cc_library(
#     name = "check_nn_recall",
#     hdrs = ["check_nn_recall.h"],
#     deps = [
#         "@parlaylib//parlay:parallel",
#         "@parlaylib//parlay:primitives",
#         ":beamSearch",
#         ":csvfile",
#         ":parse_results",
#         ":stats",
#         ":types",
#     ],
# )

cc_library(
    name = "euclidean_point",
    hdrs = ["euclidian_point.h"],
    deps = [
        "@parlaylib//parlay:parallel",
        "@parlaylib//parlay:primitives",
        "@parlaylib//parlay/internal:file_map",
        "//algorithms/bench:parse_command_line",
        ":parse_results",
        ":NSGDist",
    ],
)

cc_library(
    name = "graph",
    hdrs = ["graph.h"],
    deps = [
        "@parlaylib//parlay:parallel",
        "@parlaylib//parlay:primitives",
        "@parlaylib//parlay/internal:file_map",
        "//algorithms/bench:parse_command_line",
        ":parse_results",
        ":NSGDist",
    ],
)

cc_library(
    name = "mips_point",
    hdrs = ["mips_point.h"],
    deps = [
        "@parlaylib//parlay:parallel",
        "@parlaylib//parlay:primitives",
        "@parlaylib//parlay/internal:file_map",
        "//algorithms/bench:parse_command_line",
        ":NSGDist",
        ":types",
    ],
)

cc_library(
    name = "mmap",
    hdrs = ["mmap.h"],
    deps = [
        "@parlaylib//parlay:parallel",
        "@parlaylib//parlay:primitives",
        "@parlaylib//parlay/internal:file_map",
        "//algorithms/bench:parse_command_line",
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
    name = "parse_results",
    hdrs = ["parse_results.h"],
    deps = [
        "@parlaylib//parlay:parallel",
        "@parlaylib//parlay:primitives",
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
        ":parse_results",
        ":types",
    ],
)

cc_library(
    name = "point_range",
    hdrs = ["point_range.h"],
    deps = [
        "@parlaylib//parlay:parallel",
        "@parlaylib//parlay:primitives",
        "@parlaylib//parlay/internal:file_map",
        "//algorithms/bench:parse_command_line",
        ":types",
    ],
)

cc_library(
    name = "stats",
    hdrs = ["stats.h"],
    deps = [
        "@parlaylib//parlay:parallel",
        "@parlaylib//parlay:primitives",
    ],
)

cc_library(
    name = "types",
    hdrs = ["types.h"],
    deps = [
        "@parlaylib//parlay:parallel",
        "@parlaylib//parlay:primitives",
        ":mmap",
    ],
)


cc_library(
    name = "union",
    hdrs = ["union.h"],
    deps = [
        "@parlaylib//parlay:parallel",
        "@parlaylib//parlay:primitives",
    ],
)


