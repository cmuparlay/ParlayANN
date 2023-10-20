# g++ -DSTATS -DHOMEGROWN -pthread -mcx16 -O3 -Wall -shared -std=c++17 -march=native -DNDEBUG -I . -fPIC $(python3 -m pybind11 --includes) vamana_index.cpp -o vamana_index$(python3-config --extension-suffix) -DHOMEGROWN -pthread -ldl -L/usr/local/lib -ljemalloc 

# g++ -DSTATS -DHOMEGROWN -pthread -mcx16 -O0 -g -Wall -std=c++17 -march=native -I . -fPIC $(python3 -m pybind11 --includes) ivf_test.cpp -o ivf_test -DHOMEGROWN -pthread -ldl -L/usr/local/lib -I ../../parlaylib/include #-ljemalloc -DNDEBUG  -w

g++ -DSTATS -DHOMEGROWN -pthread -mcx16 -O3 -g -Wall -std=c++17 -march=native -I . -fPIC $(python3 -m pybind11 --includes) ivf_test.cpp -o ivf_test -DHOMEGROWN -pthread -ldl -L/usr/local/lib -I ../../parlaylib/include