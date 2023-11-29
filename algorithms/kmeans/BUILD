licenses(["notice"])

#run an instance of kmeans
cc_binary(
  name="kmeans_run",
  srcs=["kmeans.cpp"],
  linkopts=["-pthread"],
 #TODO add -03 to copts
  deps = [
    ":kmeans_headers",
    "//algorithms/utils:point_range",
    "//algorithms/bench:parse_command_line",
    ":naive",
    ":yy",

  ],
 
)

#run a benchmark across many values of n,d,k,etc.
cc_binary(
  name="bench_run",
  srcs=["bench.cpp"],
  linkopts=["-pthread"],
 
  deps = [
    ":kmeans_headers",
    "//algorithms/utils:point_range"
  ],
 
)
cc_library(
  name="init",
  hdrs=["initialization.h"],
  linkopts=["-pthread"],
  deps = [
  "@parlaylib//parlay:parallel",
  "@parlaylib//parlay:primitives",],
)
cc_library(
  name="distance",
  hdrs=["distance.h"],
  linkopts=["-pthread"],
  deps = [
  "@parlaylib//parlay:parallel",
  "@parlaylib//parlay:primitives",
 
  "//algorithms/utils:NSGDist",],

)
cc_library(
  name="kmeans_bench",
  hdrs=["kmeans_bench.h"],
  linkopts=["-pthread"],
  deps = [
  "@parlaylib//parlay:parallel",
  "@parlaylib//parlay:primitives",
  "@parlaylib//parlay:sequence",
  "@parlaylib//parlay:random",
  ],
)
cc_library(
  name="parse_files",
  hdrs=["parse_files.h",],
  linkopts=["-pthread"],
  deps=[ "@parlaylib//parlay:parallel",
  "@parlaylib//parlay:primitives",
  "@parlaylib//parlay:sequence",
  "@parlaylib//parlay:slice",
  "//algorithms/utils:point_range", #include mmap
  ],

)

cc_library(
  name="kmeans_interface",
  hdrs=["kmeans.h"],
  linkopts=["-pthread"],
  deps = [
  "@parlaylib//parlay:parallel",
  "@parlaylib//parlay:primitives",
  "@parlaylib//parlay:sequence",
  "@parlaylib//parlay:slice",
  "@parlaylib//parlay:io",
  ":kmeans_bench",
  ":distance",
  ":init",
  ":parse_files",
  "//algorithms/utils:point_range",
  ],
)
cc_library(
  name="naive",
  hdrs=["naive.h",],
  linkopts=["-pthread"],
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
  "//algorithms/utils:point_range",
  ],

)
cc_library(
  name="yy",
  hdrs=["yy.h",],
  linkopts=["-pthread"],
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
  "//algorithms/utils:point_range",

  ],

)
cc_library(
  name="kmeans_headers",
  hdrs=[],
  

linkopts=["-pthread"],
#makes it known that include is an include library
deps=[
  ":kmeans_interface",
  ":naive",
  ":yy",

  ],
  

)

cc_binary(
  name="test_run",
  srcs=["test.cpp"],
  linkopts=["-pthread"],
  #makes it known that include is an include library
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
)

#testing/exploring Google test
cc_test(
  name="test_gtest",
  size="small",
  srcs=["tests/testTests.cpp"],
  deps=["@googletest//:gtest_main",
    "kmeans_headers"],
  linkopts=["-pthread"],

)

cc_library (
  name="kmeans_test_headers",
  hdrs=["tests/kmeansTests.h"],
  linkopts=["-pthread"],
  deps=["@googletest//:gtest_main",
  "kmeans_headers",
  "//algorithms/utils:point_range",],
  
)
#testing naive
cc_test(
  name="naive_gtest",
  size="small",
  srcs=["tests/naiveTests.cpp"],
  deps=["@googletest//:gtest_main",
    "kmeans_headers",
    "//algorithms/utils:point_range",
    "kmeans_test_headers",
    ],
  linkopts=["-pthread"],
  includes=["."],

)
#testing yy
cc_test(
  name="yy_gtest",
  size="small",
  srcs=["tests/yyTests.cpp"],
  deps=["@googletest//:gtest_main",
    "kmeans_headers",
    "//algorithms/utils:point_range",
    "kmeans_test_headers",],
  linkopts=["-pthread"],
  includes=["."],

)


#comparing the msse output of naive and yy
cc_test(
  name="naive_yy_compare_gtest",
  size="small",
  srcs=["tests/naiveYyTests.cpp"],
  deps=["@googletest//:gtest_main",
    "kmeans_headers",
    "//algorithms/utils:point_range",
    "kmeans_test_headers",],
  linkopts=["-pthread"],
  includes=["."],

)
