import numpy as np
import os 
import faiss 
import sklearn.preprocessing
import time

from dataset_info import data_options

faiss_configs = {"bigann-1M": {'indexkey':"OPQ32_64,IVF65536_HNSW32,PQ64x4fsr",
    "query_args": ["nprobe=1,quantizer_efSearch=4",
                    "nprobe=2,quantizer_efSearch=4",
                    "nprobe=3,quantizer_efSearch=4",
                    "nprobe=4,quantizer_efSearch=4",
                    "nprobe=4,quantizer_efSearch=8",
                    "nprobe=5,quantizer_efSearch=8",
                    "nprobe=6,quantizer_efSearch=8",
                    "nprobe=7,quantizer_efSearch=8",
                    "nprobe=8,quantizer_efSearch=8",
                    "nprobe=10,quantizer_efSearch=8",
                    "nprobe=12,quantizer_efSearch=8",
                    "nprobe=14,quantizer_efSearch=8",
                    "nprobe=16,quantizer_efSearch=8",
                    "nprobe=20,quantizer_efSearch=32",
                    "nprobe=24,quantizer_efSearch=32",
                    "nprobe=28,quantizer_efSearch=32",
                    "nprobe=32,quantizer_efSearch=32",
                    "nprobe=64,quantizer_efSearch=64",
                    "nprobe=128,quantizer_efSearch=128",
                    "nprobe=256,quantizer_efSearch=256",
                    "nprobe=512,quantizer_efSearch=512",
                    "nprobe=1024,quantizer_efSearch=1024",]},
"bigann-10M": {'indexkey': "OPQ32_64,IVF1048576_HNSW32,PQ64x4fsr",
    "query_args": ["nprobe=1,quantizer_efSearch=4",
                "nprobe=2,quantizer_efSearch=4",
                "nprobe=4,quantizer_efSearch=4",
                "nprobe=8,quantizer_efSearch=8",
                "nprobe=16,quantizer_efSearch=16",
                "nprobe=16,quantizer_efSearch=64",
                "nprobe=32,quantizer_efSearch=32",
                "nprobe=32,quantizer_efSearch=128",
                "nprobe=64,quantizer_efSearch=16",
                "nprobe=64,quantizer_efSearch=64",
                "nprobe=64,quantizer_efSearch=256",
                "nprobe=128,quantizer_efSearch=32",
                "nprobe=128,quantizer_efSearch=128",
                "nprobe=128,quantizer_efSearch=512",
                "nprobe=256,quantizer_efSearch=64",
                "nprobe=256,quantizer_efSearch=512",
                "nprobe=512,quantizer_efSearch=512"]},
"bigann-100M": {'indexkey':"OPQ64_128,IVF1048576_HNSW32,PQ128x4fsr",
    "query_args": ["nprobe=1,quantizer_efSearch=4",
                "nprobe=2,quantizer_efSearch=4",
                "nprobe=4,quantizer_efSearch=4",
                "nprobe=8,quantizer_efSearch=4",
                "nprobe=8,quantizer_efSearch=16",
                "nprobe=16,quantizer_efSearch=16",
                "nprobe=32,quantizer_efSearch=32",
                "nprobe=64,quantizer_efSearch=64",
                "nprobe=128,quantizer_efSearch=128",
                "nprobe=128,quantizer_efSearch=512",
                "nprobe=256,quantizer_efSearch=64",
                "nprobe=256,quantizer_efSearch=512",
                "nprobe=512,quantizer_efSearch=512",
                "nprobe=1024,quantizer_efSearch=512",
                "nprobe=2048,quantizer_efSearch=512",
                "nprobe=4096,quantizer_efSearch=512"]},
"ssnpp-1M" : {},
"ssnpp-10M": {"indexkey": "OPQ32_128,IVF65536_HNSW32,PQ32"},
"ssnpp-100M": {},
}

def sanitize(x):
    """ make the simplest possible array of the input"""
    return np.ascontiguousarray(x)

'''
mmap the bin data to a numpy array
'''
def read_data(fname, dtype):
    n, d = map(int, np.fromfile(fname, dtype="uint32", count=2))
    print("Detected", n, "results with dimension", d)
    assert os.stat(fname).st_size == 8 + n * d * np.dtype(dtype).itemsize
    return np.memmap(fname, dtype=dtype, mode="r", offset=8, shape=(n, d))

def get_dataset_iterator(filename, dtype, bs=512, split=(1,0)):
    nsplit, rank = split
    x = read_data(filename, dtype)
    nb = x.shape[0]
    i0, i1 = nb * rank // nsplit, nb * (rank + 1) // nsplit
    for j0 in range(i0, i1, bs):
        j1 = min(j0 + bs, i1)
        yield sanitize(x[j0:j1])

'''
Range groundtruth consists of the total number of matches, 
an array of size n specifying how many matches each point has,
then the ids of the matches
'''
def read_groundtruth(fname):
    f = open(fname, "rb")
    nq, total_res = np.fromfile(f, count=2, dtype="int32")
    print("Detected", nq, "queries with", total_res, "total results")
    nres = np.fromfile(f, count=nq, dtype="int32")
    assert nres.sum() == total_res
    lims = np.cumsum(nres)
    lims = np.insert(lims, 0, 0, axis=0)
    assert len(lims) == nq+1
    assert lims[0] == 0
    assert lims[len(lims)-1] == total_res
    I = np.fromfile(f, count=total_res, dtype="int32")
    D = np.fromfile(f, count=total_res, dtype="float32")
    return total_res, lims, I

def translate_metric(metric):
    if metric == "Euclidian":
        return faiss.METRIC_L2
    elif metric == "mips":
        return faiss.METRIC_INNER_PRODUCT
    

def unwind_index_ivf(index):
    if isinstance(index, faiss.IndexPreTransform):
        assert index.chain.size() == 1
        vt = faiss.downcast_VectorTransform(index.chain.at(0))
        index_ivf, vt2 = unwind_index_ivf(faiss.downcast_index(index.index))
        assert vt2 is None
        return index_ivf, vt
    if hasattr(faiss, "IndexRefine") and isinstance(index, faiss.IndexRefine):
        return unwind_index_ivf(faiss.downcast_index(index.base_index))
    if isinstance(index, faiss.IndexIVF):
        return index, None
    else:
        return None, None

def two_level_clustering(xt, nc1, nc2, clustering_niter=25, spherical=False):
    d = xt.shape[1]

    print(f"2-level clustering of {xt.shape} nb clusters = {nc1}*{nc2} = {nc1*nc2}")
    print("perform coarse training")

    km = faiss.Kmeans(
        d, nc1, verbose=True, niter=clustering_niter,
        max_points_per_centroid=2000,
        spherical=spherical
    )
    km.train(xt)

    print()

    # coarse centroids
    centroids1 = km.centroids

    print("assigning the training set")
    t0 = time.time()
    _, assign1 = km.assign(xt)
    bc = np.bincount(assign1, minlength=nc1)
    print(f"done in {time.time() - t0:.2f} s. Sizes of clusters {min(bc)}-{max(bc)}")
    o = assign1.argsort()
    del km

    # train sub-clusters
    i0 = 0
    c2 = []
    t0 = time.time()
    for c1 in range(nc1):
        print(f"[{time.time() - t0:.2f} s] training sub-cluster {c1}/{nc1}\r", end="", flush=True)
        i1 = i0 + bc[c1]
        subset = o[i0:i1]
        assert np.all(assign1[subset] == c1)
        km = faiss.Kmeans(d, nc2, spherical=spherical)
        xtsub = xt[subset]
        km.train(xtsub)
        c2.append(km.centroids)
        i0 = i1
    print(f"done in {time.time() - t0:.2f} s")
    return np.vstack(c2)


class Faiss():
    def __init__(self, dataset_name):
        self.dataset = data_options[dataset_name]
        self.index_params = faiss_configs[dataset_name]
        self.metric = translate_metric(self.dataset["dist_fn"])
        self.indexkey = self.index_params['indexkey'] if 'indexkey' in self.index_params else "OPQ32_128,IVF65536_HNSW32,PQ32"

    def index_name(self, name):
        return f"data/{name}.{self.indexkey}.faissindex"

    def fit(self, dataset):
        index_params = self.index_params

        dataset_arr = read_data(self.dataset["base"], map_type(self.dataset["data_type"]))
        nb = dataset_arr.shape[0]
        d = dataset_arr.shape[1]

        # get build parameters
        buildthreads = index_params.get("buildthreads", -1)
        by_residual = index_params.get("by_residual", -1)
        maxtrain = index_params.get("maxtrain", 0)
        clustering_niter = index_params.get("clustering_niter", -1)
        add_bs = index_params.get("add_bs", 100000)
        add_splits = index_params.get("add_splits", 1)
        efSearch = index_params.get("quantizer_add_efSearch", 80)
        efConstruction = index_params.get("quantizer_efConstruction", 200)
        use_two_level_clustering = index_params.get("two_level_clustering", True)
        indexfile = self.index_name(dataset)

        if buildthreads == -1:
            print("Build-time number of threads:", faiss.omp_get_max_threads())
        else:
            print("Set build-time number of threads:", buildthreads)
            faiss.omp_set_num_threads(buildthreads)

        metric_type = self.metric 

        index = faiss.index_factory(d, self.indexkey, metric_type)

        index_ivf, vec_transform = unwind_index_ivf(index)
        if vec_transform is None:
            vec_transform = lambda x: x
        else:
            vec_transform = faiss.downcast_VectorTransform(vec_transform)

        if by_residual != -1:
            by_residual = by_residual == 1
            print("setting by_residual = ", by_residual)
            index_ivf.by_residual   # check if field exists
            index_ivf.by_residual = by_residual

        if index_ivf:
            print("Update add-time parameters")
            # adjust default parameters used at add time for quantizers
            # because otherwise the assignment is inaccurate
            quantizer = faiss.downcast_index(index_ivf.quantizer)
            if isinstance(quantizer, faiss.IndexRefine):
                print("   update quantizer k_factor=", quantizer.k_factor, end=" -> ")
                quantizer.k_factor = 32 if index_ivf.nlist < 1e6 else 64
                print(quantizer.k_factor)
                base_index = faiss.downcast_index(quantizer.base_index)
                if isinstance(base_index, faiss.IndexIVF):
                    print("   update quantizer nprobe=", base_index.nprobe, end=" -> ")
                    base_index.nprobe = (
                        16 if base_index.nlist < 1e5 else
                        32 if base_index.nlist < 4e6 else
                        64)
                    print(base_index.nprobe)
            elif isinstance(quantizer, faiss.IndexHNSW):
                print("   update quantizer efSearch=", quantizer.hnsw.efSearch, end=" -> ")
                if index_params.get("quantizer_add_efSearch", 80) > 0:
                    quantizer.hnsw.efSearch = efSearch
                else:
                    quantizer.hnsw.efSearch = 40 if index_ivf.nlist < 4e6 else 64
                print(quantizer.hnsw.efSearch)
                if efConstruction != -1:
                    print("  update quantizer efConstruction=", quantizer.hnsw.efConstruction, end=" -> ")
                    quantizer.hnsw.efConstruction = efConstruction
                    print(quantizer.hnsw.efConstruction)


        index.verbose = True
        if index_ivf:
            index_ivf.verbose = True
            index_ivf.quantizer.verbose = True
            index_ivf.cp.verbose = True


        if maxtrain == 0:
            if 'IMI' in self.indexkey:
                maxtrain = int(256 * 2 ** (np.log2(index_ivf.nlist) / 2))
            elif index_ivf:
                maxtrain = 50 * index_ivf.nlist
            else:
                # just guess...
                maxtrain = 256 * 100
            maxtrain = max(maxtrain, 256 * 100)
            print("setting maxtrain to %d" % maxtrain)

        # train on dataset
        print(f"getting first {maxtrain} dataset vectors for training")

        xt2 = next(get_dataset_iterator(self.dataset["base"], map_type(self.dataset["data_type"]), bs=maxtrain))

        print("train, size", xt2.shape)
        assert np.all(np.isfinite(xt2))

        t0 = time.time()

        if (isinstance(vec_transform, faiss.OPQMatrix) and
            isinstance(index_ivf, faiss.IndexIVFPQFastScan)):
            print("  Forcing OPQ training PQ to PQ4")
            ref_pq = index_ivf.pq
            training_pq = faiss.ProductQuantizer(
                ref_pq.d, ref_pq.M, ref_pq.nbits
            )
            vec_transform.pq
            vec_transform.pq = training_pq

        if clustering_niter >= 0:
            print(("setting nb of clustering iterations to %d" %
                    clustering_niter))
            index_ivf.cp.niter = clustering_niter

        train_index = None

        if use_two_level_clustering:
            sqrt_nlist = int(np.sqrt(index_ivf.nlist))
            assert sqrt_nlist ** 2 == index_ivf.nlist

            centroids_trainset = xt2
            if isinstance(vec_transform, faiss.VectorTransform):
                print("  training vector transform")
                vec_transform.train(xt2)
                print("  transform trainset")
                centroids_trainset = vec_transform.apply_py(centroids_trainset)

            centroids = two_level_clustering(
                centroids_trainset, sqrt_nlist, sqrt_nlist,
                spherical=(metric_type == faiss.METRIC_INNER_PRODUCT)
            )

            if not index_ivf.quantizer.is_trained:
                print("  training quantizer")
                index_ivf.quantizer.train(centroids)

            print("  add centroids to quantizer")
            index_ivf.quantizer.add(centroids)

        index.train(xt2)
        print("  Total train time %.3f s" % (time.time() - t0))

        if train_index is not None:
            del train_index
            index_ivf.clustering_index = None
            gc.collect()

        print("adding")

        t0 = time.time()
        add_bs = index_params.get("add_bs", 10000000)
        if add_bs == -1:
            index.add(ds.get_database())
        else:
            i0 = 0
            for xblock in get_dataset_iterator(self.dataset["base"], map_type(self.dataset["data_type"]), bs=add_bs):
                i1 = i0 + len(xblock)
                print("  adding %d:%d / %d [%.3f s, RSS %d kiB] " % (
                    i0, i1, nb, time.time() - t0,
                    faiss.get_mem_usage_kb()))
                index.add(xblock)
                i0 = i1

        print("  add in %.3f s" % (time.time() - t0))
        print("storing", )
        faiss.write_index(index, self.index_name(dataset))

        self.index = index
        self.ps = faiss.ParameterSpace()
        self.ps.initialize(self.index)

    def load_index(self, dataset):

        print("Loading index")

        self.index = faiss.read_index(self.index_name(dataset))

        self.ps = faiss.ParameterSpace()
        self.ps.initialize(self.index)

        return True

    def set_query_arguments(self, query_args):
        faiss.cvar.indexIVF_stats.reset()
        self.ps.set_index_parameters(self.index, query_args)
        self.qas = query_args

    def get_additional(self):
        return {"dist_comps": faiss.cvar.indexIVF_stats.ndis}

    def range_query(self, X, radius):
        self.res = self.index.range_search(X, radius)

    def get_range_results(self):
        return self.res

'''
range_result is a list of ids that lie within the range
'''
def compute_average_precision(range_result, fname):
    gt_total_res, gt_limits, gt_ids = read_groundtruth(fname)
    total_correct = 0
    reported_limits, reported_dists, reported_ids = range_result
    assert len(reported_limits) == len(gt_limits)
    for i in range(len(reported_limits)-1):
        num_correct = np.intersect1d(gt_ids[gt_limits[i]:gt_limits[i+1]], reported_ids[reported_limits[i]:reported_limits[i+1]])
        total_correct += num_correct.shape[0]
    return total_correct/gt_total_res

def map_type(dtype):
    if dtype == "uint8":
        return np.uint8
    elif dtype == "float":
        return np.float32

'''
Takes in an index object and the dataset name
Times each query, computes QPS and average precision
'''
def time_queries(dataset_name):
    index = Faiss(dataset_name)
    try:
        index.load_index(dataset_name)
    except:
        index.fit(dataset_name)
        index.load_index(dataset_name)
        
    dataset = data_options[dataset_name]
    queries = read_data(dataset["query"], map_type(dataset["data_type"]))
    radius = dataset["radius"]
    query_argument_groups = faiss_configs[dataset_name]["query_args"]
    qps = []
    ap = []
    for query_args in query_argument_groups:
        index.set_query_arguments(query_args)
        #run once for warmup
        index.range_query(queries, radius)
        start = time.time()
        index.range_query(queries, radius)
        elapsed_time = time.time() - start
        res = index.get_range_results()
        pc = compute_average_precision(res, dataset["gt"])
        ap.append(pc)
        qps_single = queries.shape[0]/elapsed_time
        qps.append(qps_single)
    print("QPS:", qps)
    print("Average Precision:", ap)
     


dataset_name = "bigann-1M"
time_queries(dataset_name)




