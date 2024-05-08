# ANNS utilility.

package(default_visibility = [
  "//algorithms:__subpackages__",
  "//rangeSearch:__subpackages__"
])


cc_library(
    name = "IO",
    hdrs = ["IO.h"],
    deps = [
        "@parlaylib//parlay:parallel",
        "@parlaylib//parlay:primitives",
        "@parlaylib//parlay:io",
        "@parlaylib//parlay/internal:get_time",
    ],
)

cc_library(
    name = "geometry",
    hdrs = ["geometry.h"],
    deps = [
        "@parlaylib//parlay:parallel",
        "@parlaylib//parlay:primitives",
    ],
)

cc_library(
    name = "geometryIO",
    hdrs = ["geometryIO.h"],
    deps = [
        ":geometry",
        ":IO",
        "@parlaylib//parlay:parallel",
        "@parlaylib//parlay:primitives",
    ],
)

cc_library(
    name = "get_time",
    hdrs = ["get_time.h"],
    deps = [],
)

cc_library(
    name = "parse_command_line",
    hdrs = ["parse_command_line.h"],
    deps = [],
)
