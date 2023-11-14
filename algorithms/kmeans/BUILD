licenses(["notice"])

#run an instance of kmeans
cc_binary(
  name="kmeans_test_run",
  srcs=["kmeans.cpp"],
  linkopts=["-pthread"],
  #makes it known that include is an include library
  copts = ["-Iinclude"],
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
  name="kmeans_headers",
  hdrs=["parse_files.h",
"initialization.h",
"naive.h",
"kmeans_bench.h",
"distance.h",
"kmeans.h",
"yy.h",
],

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

