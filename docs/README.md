# ParlayANN

ParlayANN is a library of approximate nearest neighbor search algorithms, along with a set of useful tools for designing such algorithms. It is written in C++ and uses parallel primitives from [ParlayLib](https://cmuparlay.github.io/parlaylib/). Currently it includes implementations of the ANNS algorithms [DiskANN](https://github.com/microsoft/DiskANN), [HNSW](https://github.com/nmslib/hnswlib), [HCNNG](https://github.com/jalvarm/hcnng), and [pyNNDescent](https://pynndescent.readthedocs.io/en/latest/).

To install, [clone the repo](https://github.com/cmuparlay/ParlayANN/tree/main) and then initiate the ParlayLib submodule:

```bash
git submodule init
git submodule update
```

See the following documentation for help getting started:
- [Quickstart](https://cmuparlay.github.io/ParlayANN/quickstart)
- [Algorithms](https://cmuparlay.github.io/ParlayANN/algorithms)
- [Data Tools](https://cmuparlay.github.io/ParlayANN/data_tools)

This repository was built for our paper [Scaling Graph-Based ANNS Algorithms to Billion-Size Datasets: A Comparative Analsyis](https://arxiv.org/abs/2305.04359). If you use this repository for your own work, please cite us:

```bibtex
@inproceedings{ANNScaling,
  author = {Manohar, Magdalen Dobson and Shen, Zheqi and Blelloch, Guy and Dhulipala, Laxman and Gu, Yan and Simhadri, Harsha Vardhan and Sun, Yihan},
  title = {ParlayANN: Scalable and Deterministic Parallel Graph-Based Approximate Nearest Neighbor Search Algorithms},
  year = {2024},
  isbn = {9798400704352},
  publisher = {Association for Computing Machinery},
  address = {New York, NY, USA},
  url = {https://doi.org/10.1145/3627535.3638475},
  doi = {10.1145/3627535.3638475},
  booktitle = {Proceedings of the 29th ACM SIGPLAN Annual Symposium on Principles and Practice of Parallel Programming},
  pages = {270–285},
  numpages = {16},
  keywords = {nearest neighbor search, vector search, parallel algorithms},
  location = {Edinburgh, United Kingdom},
  series = {PPoPP '24}
}
```
