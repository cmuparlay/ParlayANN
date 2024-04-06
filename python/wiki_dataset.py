# %%
import datasets
import pandas as pd
import numpy as np
from collections import Counter
from tqdm import tqdm
from tqdm.notebook import tqdm as tqdm_notebook
# from multiprocessing import Pool, Process, Queue, Manager
from scipy.sparse import csr_matrix, save_npz
import re


# %%
print(datasets.config.HF_DATASETS_CACHE)

# %%
# Load the dataset
docs = datasets.load_dataset(f"Cohere/wikipedia-22-12-en-embeddings", split="train")

# docs = docs.shard(1000, 0, contiguous=True, keep_in_memory=True)

# # %%
# N_WORKERS = 60

# shards = [docs.shard(N_WORKERS, i, contiguous=True, keep_in_memory=True) for i in range(N_WORKERS)]

# # first pass:
# # want to save the vectors to a huge numpy array, 
# # count word occurences, words lowercased
# docs[0]
# # %%
# pattern = re.compile(r'[A-Za-z]+')

# def split_alpha(text):
#     # Use the compiled regex object's findall method
#     words = pattern.findall(text.lower())
#     return words

# # %%
# # word counts
# # word_counts = Counter()
# # for doc in tqdm(docs):
# #     word_counts.update(split_alpha(doc['text']))

# # %%
# def count_shard(shard, queue):
#     word_counts = Counter()
#     # shard = docs.shard(shard_i, N_WORKERS, contiguous=True, keep_in_memory=True)
#     try:
#         for doc in tqdm(shard):
#             word_counts.update(split_alpha(doc['text']))

#         queue.put(word_counts)
#         print(f'placed count in queue {queue.qsize()} / {N_WORKERS}')
#     except Exception as e:
#         print(e)
#         queue.put(word_counts)
#     return

# def compile_shard(shard, queue, index_dict):
#     output_dict = {}
#     try:
#         for doc in tqdm(shard):
#             split_text = split_alpha(doc['text'])
#             output_dict[doc['id']] = (np.array([index_dict[word] for word in split_text if word in index_dict]), np.array(doc['emb']))
        
#         queue.put(output_dict)
#         print(f'placed shard in queue {queue.qsize()} / {N_WORKERS}')
#     except Exception as e:
#         print(e)
#         queue.put(output_dict)
#     return
    

# # %%
# if __name__ == '__main__':
#     manager = Manager()
#     output_queue = manager.Queue()
#     processes = []
#     for shard in shards:
#         p = Process(target=count_shard, args=(shard, output_queue))
#         p.start()
#         processes.append(p)

#     print('done starting processes')

#     for p in processes:
#         p.join()

#     print('done counting words')

#     word_counts = Counter()
#     for _ in tqdm(range(N_WORKERS)):
#         word_counts.update(output_queue.get())
    
#     print('done combining word counts')
    
#     # filter out words that occur less than 5 times
#     word_counts = Counter({word: count for word, count in word_counts.items() if count >= 5})

#     print('done filtering on word count')

#     # writing the word counts to a file just in case
#     with open('wiki_word_counts_trunc.txt', 'w') as f:
#         f.writelines((f'{word} {count}\n' for word, count in word_counts.most_common()))

#     print('done writing word counts to file')

#     # dict from word to index
#     word_index = {word: i for i, (word, _) in enumerate(word_counts.most_common())}

#     print('done creating word index')

#     # use multiprocessing to compile the shards
#     output_queue = manager.Queue()
#     processes = []
#     for shard in shards:
#         p = Process(target=compile_shard, args=(shard, output_queue, word_index))
#         p.start()
#         processes.append(p)

#     print('done starting processes')

#     for p in processes:
#         p.join()

#     print('done compiling shards')

#     vectors = np.zeros((len(docs), len(docs[0]["emb"])), dtype=np.float32)
#     metadata_list = [None] * len(docs)

#     for output_dict in tqdm((output_queue.get() for _ in range(N_WORKERS)), total=N_WORKERS):
#         for i, (indices, emb) in output_dict.items():
#             vectors[i, :] = emb
#             metadata_list[i] = indices

#     print('done compiling vectors and metadata')

#     # initialize the sparse matrix
#     nnz = sum(word_counts.values())
#     metadata_offsets = np.zeros(len(docs) + 1, dtype=np.int64)
#     metadata_indices = np.zeros(nnz, dtype=np.int32)

#     print('done initializing sparse matrix')
#     for i, indices in enumerate(tqdm(metadata_list)):
#         metadata_offsets[i + 1] = metadata_offsets[i] + len(indices)
#         if len(indices) > 0:
#             indices.sort()
#             metadata_indices[metadata_offsets[i] : metadata_offsets[i + 1]] = indices

#     metadata = csr_matrix((np.ones(nnz), metadata_indices, metadata_offsets), dtype=np.int8)
#     print('done filling sparse matrix')

#     # save the vectors and metadata
#     with open(f"/ssd1/anndata/wiki_sentence/base.{len(docs)}.fbin", mode='wb') as vector_file:
#         vector_file.write(np.array(vectors.shape, dtype=np.int32).tobytes())
#         vectors.tofile(vector_file)

#     save_npz(f"/ssd1/anndata/wiki_sentence/base.metadata.{len(docs)}.spmat", metadata)

#     print('done saving vectors and metadata')



# %%
# same as above but with multiprocessing
# from multiprocessing import Pool
# from tqdm import tqdm

# def count_words(doc):
#     return Counter(doc['text'].lower().split(' '))

# with Pool(164) as p:
#     word_counts = Counter()
#     for counts in tqdm(p.imap_unordered(count_words, docs, chunksize=10_000), total=len(docs)):
#         word_counts.update(counts)

# %%

# %%
# load the word counts from file
word_counts = Counter()
with open('wiki_word_counts.txt', 'r') as f:
    for line in f:
        word, count = line.split(' ')
        word_counts[word] = int(count)

# restricting to nouns is going to make ands too selective
# %%
# # we're restricting to nouns, so we load nouns from nltk
# from nltk.corpus import wordnet as wn
# import nltk

# # nltk.download('wordnet')
# # %%
# nouns = {synset.name().split('.')[0] for synset in wn.all_synsets('n')}
# print(len(nouns))
# print(nouns)

# single_word_nouns = {word for word in nouns if '_' not in word}
# print(len(single_word_nouns))
# print(single_word_nouns)

# # restrict word counts to single word nouns
# word_counts = Counter({word: count for word, count in word_counts.items() if word in single_word_nouns})
# print(len(word_counts))

# %%
KEPT_WORDS = 4_000
# dict from word to index
word_index = {word: i for i, (word, _) in enumerate(word_counts.most_common()[:KEPT_WORDS])}


# %%
def write_csr(matrix, filename):
    with open(filename, 'wb') as f:
        np.array([matrix.shape[0], matrix.shape[1], matrix.nnz], dtype=np.int64).tofile(f)
        matrix.indptr.astype(np.int64).tofile(f)
        matrix.indices.astype(np.int32).tofile(f)
        matrix.data.astype(np.int32).tofile(f)

def numpy_to_fvec(data, outfile):
    with open(outfile, 'wb') as f:
        np.array([data.shape[0]], dtype=np.int32).tofile(f)
        np.array([data.shape[1]], dtype=np.int32).tofile(f)
        data.astype(np.float32).tofile(f)
# %%
# # we want to reserve 100k vectors for queries
import random

# # set seed for reproducibility
# np.random.seed(0)
# query_indices = np.array(sample(range(len(docs)), 100_000))

# # %%
# # we compute the metadata sparse matrix and the vectors dense matrix for the queries
# q_metadata_offsets = np.zeros(len(query_indices) + 1, dtype=np.int64)
# q_metadata_indices = np.zeros(sum(word_counts.values()), dtype=np.int32) # will need to be shortened to last value of q_metadata_offsets
# q_vectors = np.zeros((len(query_indices), len(docs[0]["emb"])), dtype=np.float32)

# for i, doc_i in enumerate(tqdm(query_indices)):
#     doc = docs[int(doc_i)]
#     # set the vector
#     q_vectors[i, :] = doc["emb"]
#     # set the metadata
#     split_text = doc["text"].lower().split(" ")
#     surviving_words = [word for word in split_text if word in word_index]

#     if len(surviving_words) == 0:
#         labels = ['the']
#     elif len(surviving_words) == 1:
#         labels = surviving_words
#     else:
#         labels = sample(surviving_words, 2)

#         if labels[0] == labels[1]:
#             labels = labels[:1]

#     q_metadata_offsets[i+1] = q_metadata_offsets[i] + len(labels)
#     q_metadata_indices[q_metadata_offsets[i] : q_metadata_offsets[i + 1]] = [word_index[word] for word in labels]

# q_metadata_indices = q_metadata_indices[:q_metadata_offsets[-1]]

# query_csr = csr_matrix((np.ones(len(q_metadata_indices)), q_metadata_indices, q_metadata_offsets), dtype=np.float32)

# write_csr(query_csr, "/ssd1/anndata/wiki_sentence/query.metadata.spmat")
# numpy_to_fvec(q_vectors, "/ssd1/anndata/wiki_sentence/query.fbin")


# %% 
# initialize a sparse matrix for the labels
# and a dense matrix for the vectors
from scipy.sparse import csr_matrix, save_npz
from scipy.sparse import vstack as sparse_vstack

# %%
# # initialize the sparse matrix
# # metadata = csr_matrix((len(docs), len(word_counts)), dtype=np.int8)
# metadata_offsets = np.zeros(len(docs) - len(query_indices) + 1, dtype=np.int64)
# metadata_indices = np.zeros(sum(word_counts.values()) - query_csr.nnz, dtype=np.int32)
# vectors = np.zeros((len(docs) - len(query_indices), len(docs[0]["emb"])), dtype=np.float32)

# # q_metadata_offsets = np.zeros(len(query_indices) + 1, dtype=np.int64)


# # %%

# for i, doc_i in enumerate(tqdm(np.setdiff1d(range(len(docs)), query_indices))):
#     doc = docs[int(doc_i)]

#     # set the vector
#     vectors[i, :] = doc["emb"]
#     # set the metadata
#     split_text = doc["text"].lower().split(" ")
#     surviving_words = [word for word in split_text if word in word_index]
#     metadata_offsets[i + 1] = metadata_offsets[i] + len(surviving_words)
#     if len(surviving_words) > 0:
#         metadata_indices[metadata_offsets[i] : metadata_offsets[i + 1]] = [
#             word_index[word] for word in surviving_words
#         ]
    
# metadata = csr_matrix((np.ones(len(metadata_indices)), metadata_indices, metadata_offsets), dtype=np.int8)

# # %%
    
# write_csr(metadata, "/ssd1/anndata/wiki_sentence/base.metadata.spmat")

# print(f"metadata shape: {metadata.shape}, metadata nnz: {metadata.nnz}")

# # %%
# numpy_to_fvec(vectors, "/ssd1/anndata/wiki_sentence/base.fbin")


# %%
# from random import choices # we'll sample with replacement, and if both values are the same, it's a one filter query

# # need to generate queries
# q_docs = datasets.load_dataset(f"Cohere/wikipedia-22-12-en-embeddings", split="validation")

# print(len(q_docs))

# q_metadata_offsets = np.zeros(len(q_docs))
# q_metadata_indices = []
# q_vectors = np.zeros((len(q_docs), len(docs[0]["emb"])), dtype=np.float32)

# for i, doc in tqdm(enumerate(q_docs), total=len(q_docs)):
#     # set the vector
#     q_vectors[i, :] = doc["emb"]
#     # set the metadata
#     split_text = doc["text"].lower().split(" ")
#     surviving_words = [word for word in split_text if word in word_index]

#     if len(surviving_words) == 0:
#         labels = ['the']

#     labels = choices(surviving_words, 2)
#     if labels[0] == labels[1]:
#         labels = labels[:1]

#     q_metadata_offsets[i] = len(q_metadata_indices)
#     q_metadata_indices.extend([word_index[word] for word in labels])
    
# q_metadata_indices = np.array(q_metadata_indices)

# q_metadata = csr_matrix((np.ones(len(q_metadata_indices)), q_metadata_indices, q_metadata_offsets), dtype=np.int8)

# # %%
# write_csr(q_metadata, "/ssd1/anndata/wiki_sentence/query.metadata.spmat")
# numpy_to_fvec(q_vectors, "/ssd1/anndata/wiki_sentence/query.fbin")

# print(f"query metadata shape: {q_metadata.shape}, query metadata nnz: {q_metadata.nnz}")
# # %%

# %%

# %%
# we need to save the first 35M vectors and put together a 4k word metadata matrix
N = 35_000_000
vectors = np.zeros((N, len(docs[0]["emb"])), dtype=np.float32)
metadata_offsets = []
metadata_indices = []

for i, doc in tqdm(enumerate(docs), total=N):
    if i >= N:
        break
    # set the vector
    vectors[i, :] = doc["emb"]
    # set the metadata
    split_text = doc["text"].lower().split(" ")
    surviving_word_indices = list(set([word_index[word] for word in split_text if word in word_index]))
    surviving_word_indices.sort()
    metadata_offsets.append(len(metadata_indices))
    metadata_indices.extend(surviving_word_indices)

metadata_offsets.append(len(metadata_indices))

metadata_offsets = np.array(metadata_offsets, dtype=np.int64)
metadata_indices = np.array(metadata_indices, dtype=np.int32)

metadata = csr_matrix((np.ones(len(metadata_indices)), metadata_indices, metadata_offsets), dtype=np.int8)

write_csr(metadata, "/ssd1/anndata/wiki_sentence/base.35M.4k.metadata.spmat")
numpy_to_fvec(vectors, "/ssd1/anndata/wiki_sentence/base.35M.4k.fbin")

# %%
# same as above but sampling 1_000_000 vectors
N = 1_000_000
vectors = np.zeros((N, len(docs[0]["emb"])), dtype=np.float32)
metadata_offsets = []
metadata_indices = []

random.seed(0) # for reproducibility
included_indices = list(random.sample(range(len(docs)), N))

for i, index in tqdm(enumerate(included_indices), total=N):
    doc = docs[index]
    # set the vector
    vectors[i, :] = doc["emb"]
    # set the metadata
    split_text = doc["text"].lower().split(" ")
    surviving_word_indices = [word_index[word] for word in split_text if word in word_index]
    metadata_offsets.append(len(metadata_indices))
    metadata_indices.extend(surviving_word_indices)

metadata_offsets.append(len(metadata_indices))

metadata_offsets = np.array(metadata_offsets, dtype=np.int64)
metadata_indices = np.array(metadata_indices, dtype=np.int32)

metadata = csr_matrix((np.ones(len(metadata_indices)), metadata_indices, metadata_offsets), dtype=np.int8)

write_csr(metadata, "/ssd1/anndata/wiki_sentence/base.1M.4k.metadata.spmat")
numpy_to_fvec(vectors, "/ssd1/anndata/wiki_sentence/base.1M.4k.fbin")

# %%
