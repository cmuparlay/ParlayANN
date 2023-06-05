# ParlayANN

ParlayANN is a library of approximate nearest neighbor search algorithms, along with a set of useful tools for designing such algorithms. It is written in C++ and uses parallel primitives from [ParlayLib](https://cmuparlay.github.io/parlaylib/). Currently it includes implementations of the ANNS algorithms [DiskANN](https://github.com/microsoft/DiskANN), [HCNNG](https://github.com/jalvarm/hcnng), and [pyNNDescent](https://pynndescent.readthedocs.io/en/latest/).

To install, git clone and then initiate the ParlayLib submodule:

```bash
git submodule init
git submodule update
```

See the following documentation for help getting started:
- [Quickstart](https://magdalendobson.github.io/ParlayANN/quickstart)
- [Algorithms](https://magdalendobson.github.io/ParlayANN/algorithms)
- [Data Tools](https://magdalendobson.github.io/ParlayANN/data_tools)