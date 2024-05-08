# ANNS utilility.

package(default_visibility = [
  "//:__subpackages__",
])


cc_library(
    name = "benchUtils",
    hdrs = ["benchUtils.h"],
    deps = [
        "@parlaylib//parlay:parallel",
        "@parlaylib//parlay:primitives",
        "//algorithms/bench/common:geometry",
        "//algorithms/bench/common:geometryIO",
        "//algorithms/bench/common:parse_command_line",
    ],
)

cc_library(
    name = "get_time",
    hdrs = ["get_time.h"],
    deps = [],
)

cc_library(
    name = "IO",
    hdrs = ["IO.h"],
    deps = [
        "@parlaylib//parlay:io",
        "@parlaylib//parlay:parallel",
        "@parlaylib//parlay:primitives",
        "@parlaylib//parlay/internal:get_time",
    ],
)

cc_library(
    name = "parse_command_line",
    hdrs = ["parse_command_line.h"],
    deps = [],
)

cc_library(
    name = "time_loop",
    hdrs = ["time_loop.h"],
    deps = [":get_time"],
)




