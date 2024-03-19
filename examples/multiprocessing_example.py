import argparse
from multiprocessing import Pool
import os
import time

from langchain_community.embeddings import HuggingFaceEmbeddings
from langchain_community.vectorstores import FAISS
from langchain_text_splitters import RecursiveCharacterTextSplitter
import numpy as np

from langchain_progress import ProgressManager, MultiprocessingPBar


def process_shard(shard, process_idx, pbar):
    start_time = time.time()
    embeddings = HuggingFaceEmbeddings(
        model_name='sentence-transformers/all-MiniLM-L6-v2',
        model_kwargs={'device': 'cpu'},
    )
    with ProgressManager(embeddings, pbar):
        result = FAISS.from_documents(shard, embeddings)

    print(f'Shard {process_idx} processed in {time.time() - start_time:.2f} seconds.')
    return result


def main(args):
    # Fetch and process Emma by Jane Austen from Project Gutenberg
    os.system('wget https://www.gutenberg.org/cache/epub/158/pg158.txt')

    with open('pg158.txt', 'r') as file:
        doc = file.read().strip()

    os.remove('pg158.txt')

    text_splitter = RecursiveCharacterTextSplitter()
    docs = text_splitter.create_documents([doc])

    doc_shards = np.array_split(docs, args.num_shards)

    with MultiprocessingPBar(total=len(docs)) as pbar, Pool(args.num_shards) as pool:
        vectors = pool.starmap(process_shard, [(shard, idx, pbar) for idx, shard in enumerate(doc_shards)])

    vector_store = vectors[0]
    for vectors_i in vectors[1:]:
        vector_store.merge_from(vectors_i)

    # Do something with your vector store


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description='Multiprocessing example for langchain-progress ProgressManager.')
    parser.add_argument('--num-shards', type=int, default=4, help='Number of shards to split the '
                        ' documents into (default: %(default)s).')
    args = parser.parse_args()
    main(args)