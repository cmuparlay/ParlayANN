licenses(["notice"])

#run an instance of kmeans
cc_binary(
  name="kmeans_test_run",
  srcs=["kmeans.cpp"],
  linkopts=["-pthread"],
  #makes it known that include is an include library
  copts = ["-Iinclude"], #TODO add -03
  deps = [
    ":kmeans_headers",
    "//algorithms/utils:point_range"
    
    
  ],
 
)

#run a benchmark across many values of n,d,k,etc.
cc_binary(
  name="bench_run",
  srcs=["bench.cpp"],
  linkopts=["-pthread"],
  #makes it known that include is an include library
  copts = ["-Iinclude"],
  deps = [
    ":kmeans_headers",
    "//algorithms/utils:point_range"
  ],
 
)
cc_library(
  name="init",
  hdrs=["initialization.h"],
  linkopts=["-pthread"],
  copts=["-Iinclude"],
  deps = [
  "@parlaylib//parlay:parallel",
  "@parlaylib//parlay:primitives",
  "@parlaylib//parlay:sequence",
  "@parlaylib//parlay:slice",
  "@parlaylib//parlay:io",]
)
cc_library(
  name="distance",
  hdrs=["distance.h"],
  linkopts=["-pthread"],
  copts=["-Iinclude"],
  deps = [
  "@parlaylib//parlay:parallel",
  "@parlaylib//parlay:primitives",
  "@parlaylib//parlay:sequence",
  "@parlaylib//parlay:slice",
  "@parlaylib//parlay:io",
  "//algorithms/utils:NSGDist",
  "//algorithms/bench:parse_command_line",],

)
cc_library(
  name="kmeans_bench",
  hdrs=["kmeans_bench.h"],
  linkopts=["-pthread"],
  copts=["-Iinclude"],
  deps = [
  "@parlaylib//parlay:parallel",
  "@parlaylib//parlay:primitives",
  "@parlaylib//parlay:sequence",
  "@parlaylib//parlay:slice",
  "@parlaylib//parlay:io",
  "//algorithms/utils:NSGDist",
  "//algorithms/bench:parse_command_line",
  ],
)
cc_library(
  name="kmeans_interface",
  hdrs=["kmeans.h"],
  linkopts=["-pthread"],
  copts=["-Iinclude"],
  deps = [
  "@parlaylib//parlay:parallel",
  "@parlaylib//parlay:primitives",
  "@parlaylib//parlay:sequence",
  "@parlaylib//parlay:slice",
  "@parlaylib//parlay:io",
  ":kmeans_bench",
  ":distance",
  ":init",
  ],
)
cc_library(
  name="naive",
  hdrs=["naive.h",],
  linkopts=["-pthread"],
  copts=["-Iinclude"],
  deps = [
  "@parlaylib//parlay:parallel",
  "@parlaylib//parlay:primitives",
  "@parlaylib//parlay:sequence",
  "@parlaylib//parlay:slice",
  "@parlaylib//parlay:io",
  ":kmeans_bench",
  ":distance",
  ":init",
  ":kmeans_interface",
  ],

)
cc_library(
  name="yy",
  hdrs=["yy.h",],
  linkopts=["-pthread"],
  copts=["-Iinclude"],
  deps = [
  "@parlaylib//parlay:parallel",
  "@parlaylib//parlay:primitives",
  "@parlaylib//parlay:sequence",
  "@parlaylib//parlay:slice",
  "@parlaylib//parlay:io",
  ":kmeans_bench",
  ":distance",
  ":init",
  ":kmeans_interface",
  ":naive",
  "//algorithms/bench:parse_command_line",
  ],

)
cc_library(
  name="kmeans_headers",
  hdrs=["parse_files.h",],
  

linkopts=["-pthread"],
#makes it known that include is an include library
copts = ["-Iinclude"],
deps = [
  "@parlaylib//parlay:parallel",
  "@parlaylib//parlay:primitives",
  "@parlaylib//parlay:sequence",
  "@parlaylib//parlay:slice",
  "@parlaylib//parlay:io",
  "//algorithms/utils:NSGDist",
  "//algorithms/bench:parse_command_line",
  ":yy",
],

)

cc_binary(
  name="test_run",
  srcs=["test.cpp"],
  linkopts=["-pthread"],
  #makes it known that include is an include library
  copts = ["-Iinclude"],
  deps = [
    ":kmeans_headers",
    "//algorithms/utils:point_range",
    
  ],
 
)

#testing the distance function
cc_test(
  name="distance_gtest",
  size="small",
  srcs=["tests/distanceTests.cpp"],
  deps=["@googletest//:gtest_main",
    "kmeans_headers"],
  linkopts=["-pthread"],
  copts = ["-Iinclude"],
)

#testing/exploring Google test
cc_test(
  name="test_gtest",
  size="small",
  srcs=["tests/testTests.cpp"],
  deps=["@googletest//:gtest_main",
    "kmeans_headers"],
  linkopts=["-pthread"],
  copts = ["-Iinclude"],

)

#testing naive
cc_test(
  name="naive_gtest",
  size="small",
  srcs=["tests/naiveTests.cpp"],
  deps=["@googletest//:gtest_main",
    "kmeans_headers",
    "//algorithms/utils:point_range",],
  linkopts=["-pthread"],
  includes=["."],

)

