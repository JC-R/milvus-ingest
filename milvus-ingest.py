"""
This example is runnable for Milvus(0.11.x) and pymilvus(0.3.x).
"""

import random
import csv
from pprint import pprint
from time import process_time

from milvus import Milvus, DataType
import zipfile as z
import numpy as np
import argparse
import sys

class Ingester:

    def __init__(self, host, port, collection, collection_param, partition=None, drop=False, batch_size=100, dtype=np.float32):
        self.collection = collection
        self.partition = partition
        self.client = Milvus(host, port)
        self.dtype = dtype
        self.batch_size = batch_size
        if drop and self.client.has_collection(collection):
            self.client.drop_collection(collection)
        if not self.client.has_collection(collection):
            self.client.create_collection(collection, collection_param)
        if partition and not self.client.has_partition(collection, partition):
            self.client.create_partition(collection, partition)

    def ingest(self, entities, ids):
        if self.partition:
            return self.client.insert(self.collection, entities, ids=ids, partition_tag=self.partition)
        else:
            return self.client.insert(self.collection, entities, ids=ids)


if __name__ == "__main__":

    parser = argparse.ArgumentParser("Ingest embeddings into a milvus server")
    parser.add_argument("-i", "--input", type=str, required=True, help="Input embedgdings in numpy archive format, one doc per file")
    parser.add_argument("--host", type=str, default='localhost')
    parser.add_argument("--port", type=int, default=19530)
    parser.add_argument("-c", "--collection", type=str, required=True)
    parser.add_argument("--partition", type=str)
    parser.add_argument("-d", "--dimensions", type=int, required=True, help="Number of vector dimentions")
    parser.add_argument("-b", "--batchsize", type=int, default=1000, help="number of vectors per ingest")
    parser.add_argument("-a", "--append", action="store_true", help="append to the collection/map file")
    parser.add_argument("-o", "--output", type=str, required=True, help="Map file, from vilmus ID to doc ID")
    parser.add_argument("--test", action="store_true", help="Run a random search after ingestion")

    args = parser.parse_args()

    collection_param = {
        "fields": [
            {"name": "embedding", "type": DataType.FLOAT_VECTOR, "params": {"dim": 768}},
        ],
        "segment_row_limit": 4096,
        "auto_id": False
    }

    milvus = Ingester(args.host,
                      args.port,
                      args.collection,
                      collection_param,
                      args.partition,
                      drop=not args.append,
                      batch_size=args.batchsize,
                      dtype=np.float32)
    client = milvus.client

    # open the input zip
    # fname = 'D:\Downloads\cw09b.dh.k100.tier.50.t5-base.embeddings.npz'

    embeddings = []
    ids = []
    idx = 0
    map = {}
    batch = 0
    entities = [
        {"name": "embedding", "type": DataType.FLOAT_VECTOR, "values": embeddings},
    ]


    if args.append:
        sys.stderr.write("\nAppending {} to collection: {}".format(args.input, args.collection))
        fout = open(args.output, 'a')
    else:
        fout = open(args.output, 'w')

    # input is numpy filestore compressed, one doc per file
    t0 = process_time()
    with np.load(args.input, mmap_mode='r') as data:
        for fname, f in zip(data.files, data.zip.filelist):
            v = np.frombuffer(data.zip.read(f), dtype=milvus.dtype).astype(float)
            embeddings.append(v.tolist())
            ids.append(idx)
            fout.write("{},{}\n".format(idx,fname))
            map[idx] = fname
            idx += 1
            batch += 1
            if batch >= milvus.batch_size:
                idl = milvus.ingest(entities, ids)
                batch = 0
                embeddings.clear()
                ids.clear()
                sys.stderr.write("\r{} vectors ...".format(idx))
        if batch>0:
            idl = milvus.ingest(entities, ids)
            batch = 0
    if batch>0:
        milvus.ingest(entities,ids)
        batch = 0
    milvus.client.flush()

    sys.stderr.write("\rIngested {} vectors in {} seconds".format(idx, process_time()-t0))
    after_flush_counts = client.count_entities(milvus.collection)
    sys.stderr.write(" > There are {} embeddings in collection `{}` after flush\n".format(after_flush_counts, milvus.collection))

    if args.test:
        # ------
        # Basic hybrid search entities; check operating correctly
        # -----
        #
        save_embedding = None
        sys.stderr.write("\nTesting...")

        # ------
        # Basic hybrid search entities:
        #     If we want to use index, the specific index params need to be provided, in our case, the "params"
        #     should be "nprobe", if no "params" given, Milvus will complain about it and raise a exception.
        # ------
        client.create_index(milvus.collection, "embedding",
                            {"index_type": "IVF_FLAT", "metric_type": "L2", "params": {"nlist": 100}})

        query_embedding = [random.random() for _ in range(768)]


        class Query:
            def __init__(self, client, collection):
                self.client = client
                self.collection = collection

            def searh_vec(self, embedding, k, metric, nprobe):
                query = {"bool": {"must": [{"vector": {"embedding":
                                                           {"topk": k, "query": [embedding], "metric_type": metric,
                                                            "params": {"nprobe": nprobe}}}}]}}
                return self.client.search(self.collection, query, fields=["embedding"])

            def search_id(self, id, k, metric, nprobe):
                query = {"bool": {"must": [{"term": {"":
                                                         {"topk": k, "query": [embedding], "metric_type": metric,
                                                          "params": {"nprobe": nprobe}}}}]}}
                return self.client.search(self.collection, query, fields=["embedding"])


        sys.stderr.write("Random vector knn L2 search...\n")
        t1 = process_time()
        results = Query(milvus.client,milvus.collection).searh_vec(query_embedding, 3, "L2", 8)
        t2 = process_time()
        sys.stderr.write("Search took {} seconds\n".format(t2-t1))
        for entities in results:
            for topk in entities:
                if not save_embedding:
                    save_embedding = topk.entity.get("embedding")
                    save_id = topk.entity.id
                sys.stderr.write("id {} - doc {} - distance {}\n".format(topk.entity.id, map[topk.entity.id], topk.distance))

        sys.stderr.write("\n")
        sys.stderr.write("testing a specific document for knn L2 search... doc {}\n".format(map[save_id]))
        t1 = process_time()
        results = Query(milvus.client,milvus.collection).searh_vec(save_embedding, 3, "L2", 8)
        t2 = process_time()
        sys.stderr.write("Search took {} seconds\n".format(t2-t1))
        for entities in results:
            for topk in entities:
                sys.stderr.write("id {} - doc {} - distance {}\n".format(topk.entity.id, map[topk.entity.id], topk.distance))

        sys.stderr.write("\n")
