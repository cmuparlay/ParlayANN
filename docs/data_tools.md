# Data Tools

ParlayANN provides various useful tools for manipulating and reading datasets in common formats. For all of the examples below, it is assumed that the BIGANN dataset is downloaded and stored in ParlayANN/data/sift. You can do this using the following commandline:

```bash
mkdir -p data && cd data
wget ftp://ftp.irisa.fr/local/texmex/corpus/sift.tar.gz
tar -xf sift.tar.gz
```

## Compute Groundtruth

ParlayANN supports computing the exact groundtruth for k-nearest neighbors for bin files and fvecs files. The commandline for computing the groundtruth takes the following parameters:
1. **-base_path**: pointer to the base file, which ground truth will be calculate with respect to.
2. **-query_path**: pointer to the query file, for which the ground truth will be calculated.
3. **-data_type**: type of the query and base files. Current options are "uint8", "int8", and "float".
4. **-k**: the number of nearest neighbors to calculate. Default is 100.
5. **-dist_func**: the distance function to use when computing the ground truth. Current options are "euclidian" for Euclidian distance and "mips" for maximum inner product.
6. **-gt_path**: the path where the new groundtruth file will be written

The following is an example of how to compute the groundtruth for a 100K slice of the BIGANN dataset:

```bash
make compute_groundtruth
./compute_groundtruth -base_path ../data/sift/sift_learn.fbin -query_path ../data/sift/sift_query.fbin -data_type float -k 100 -dist_func Euclidian -gt_path ../data/sift/sift-100K
```

## File Conversion

ParlayANN supports converting a .vecs file to a .bin file for vectors with `float`, `uint8`, and `int` coordinates. An example commandline:

```bash
make vec_to_bin
./vec_to_bin float ../data/sift/sift_learn.fvecs ../data/sift/sift_learn.fbin
```

## Cropping

Crop a file to the desired size:

```bash
make crop
./crop ../../data/sift/sift_learn.fbin 50000 float ../data/sift/sift_50K.fbin
```

## Random Sampling

Take a random sample of desired size from a file:

```bash
make random_sample

```


