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
3. **-file_type**: currently, .bin files and .vecs files are supported. Use "bin" for .bin files and "vec" for .vecs files.
4. **-data_type**: type of the query and base files. Current options are "uint8", "int8", and "float".
5. **-k**: the number of nearest neighbors to calculate. Default is 100.
6. **-dist_func**: the distance function to use when computing the ground truth. Current options are "euclidian" for Euclidian distance and "mips" for maximum inner product.
7. **-gt_path**: the path where the new groundtruth file will be written

The following is an example of how to compute the groundtruth for a 100K slice of the BIGANN dataset:

```bash
make compute_groundtruth
./compute_groundtruth -base_path ../data/sift/sift_learn.fvecs -query_path ../data/sift/sift_query.fvecs -file_type vec -data_type float -k 100 -dist_func Euclidian ../data/sift/sift-100K
```

## File Conversion

ParlayANN supports converting a .vecs file to a .bin file, currently only for vectors with uint8 coordinates. It takes the following arguments:
1. The .bvecs file to convert.
2. Pointer to a file where the newly converted uint8bin file will be stored.
