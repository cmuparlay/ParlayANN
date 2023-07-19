# ParlayANN

ParlayANN is a library of approximate nearest neighbor search algorithms, along with a set of useful tools for designing such algorithms. It is written in C++ and uses parallel primitives from [ParlayLib](https://cmuparlay.github.io/parlaylib/). Currently it includes implementations of the ANNS algorithms [DiskANN](https://github.com/microsoft/DiskANN), [HCNNG](https://github.com/jalvarm/hcnng), and [pyNNDescent](https://pynndescent.readthedocs.io/en/latest/).

To install, [clone the repo](https://github.com/magdalendobson/ParlayANN/tree/main) and then initiate the ParlayLib submodule:

```bash
git submodule init
git submodule update
```

See the following documentation for help getting started:
- [Quickstart](https://magdalendobson.github.io/ParlayANN/quickstart)
- [Algorithms](https://magdalendobson.github.io/ParlayANN/algorithms)
- [Data Tools](https://magdalendobson.github.io/ParlayANN/data_tools)

This repository was built for our paper [Scaling Graph-Based ANNS Algorithms to Billion-Size Datasets: A Comparative Analsyis](https://arxiv.org/abs/2305.04359). If you use this repository for your own work, please cite us:

```bibtex
@article{ANNScaling,
  author       = {Magdalen Dobson and
                  Zheqi Shen and
                  Guy E. Blelloch and
                  Laxman Dhulipala and
                  Yan Gu and
                  Harsha Vardhan Simhadri and
                  Yihan Sun},
  title        = {Scaling Graph-Based {ANNS} Algorithms to Billion-Size Datasets: {A}
                  Comparative Analysis},
  journal      = {CoRR},
  volume       = {abs/2305.04359},
  year         = {2023},
  url          = {https://doi.org/10.48550/arXiv.2305.04359},
  doi          = {10.48550/arXiv.2305.04359},
  eprinttype    = {arXiv},
  eprint       = {2305.04359}
}
```
