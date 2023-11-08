licenses(["notice"])

#What's with this binary library separation?
cc_binary(
  name="kmeans_test_run",
  srcs=["kmeans.cpp"],
  linkopts=["-pthread"],
  #makes it known that include is an include library
  copts = ["-Iinclude"],
  deps = [
    ":kmeans_headers",
    
  ],
 
)

cc_library(
  name="kmeans_headers",
  hdrs=["parse_files.h",
"initialization.h",
"naive.h",
"kmeans_bench.h",
"distance.h",
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
