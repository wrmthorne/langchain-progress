# Langchain Progress

A module that adds a context manager to wrap lanchain embedding elements to better handle progress bars. This is particularly useful when using ray or multiprocessing to use a single progress bar across all remotes/processes


## Installing

The library can be installed using PyPI:

```bash
pip install langchain-progress
```

If you only need a subset of the library's features, you can install dependencies for your chosen setup:

```bash
pip install langchain-progress[tqdm]
pip install langchain-progress[ray]
```

## How to Use

This context manager can be used in a single-process or across a distributed process such as ray to display the process of generating embeddings using langchain. The ProgressManager context manager requires that a langchain embedding object be provided and optionally accepts a progress bar. If no progress bar is provided, a new progress bar will be created using tqdm. An important note is that if using `show_progress=True` when instantiating an embeddings object, any internal progress bar created within that class will be replaced with one from langchain-progress.

The following is a simple example of passing an existing progress bar and depending on the automatically generated progress bar.

```python
from langchain_progress import ProgressManager

with ProgressManager(embeddings):
    result = FAISS.from_documents(docs, embeddings)

with ProgressManager(embeddings, pbar):
    result = FAISS.from_documents(docs, embeddings)
```

### Ray Example

The real use-case for this context manager is when using ray or multiprocessing to improve embedding speed. If `show_progress=True` is enabled for embeddings objects, a new  progress bar is created for each process. This causes fighting while drawing each individual progress bar, causing the progress bar to be redrawn for each update on each process. This approach also doesn't allow us to report to a single progress bar across all remotes for a unified indication of progress. Using the `ProgressManager` context manager we can solve these problems. We can also use the `RayPBar` context manager to simplify the setup and passing of ray progress bars. The following is the recommended way to create progress bars using ray:

```python
from ray.experimental import tqdm_ray

from langchain_progress import RayPBar

@ray.remote(num_gpus=1)
def process_shard(shard, pbar):
    embeddings = HuggingFaceEmbeddings(model_name='sentence-transformers/all-MiniLM-L6-v2')

    with ProgressManager(embeddings, pbar):
        result = FAISS.from_documents(shard, embeddings)

    return result

doc_shards = np.array_split(docs, num_shards)

with RayPBar(total=len(docs)) as pbar:
    vectors = ray.get([process_shard.remote(shard, pbar) for shard in doc_shards])

pbar.close.remote()
```

A full example can be found in `./examples/ray_example.py`.

### Multiprocessing Example

To simplify implementing progress bars with multiprocessing, the `MultiprocessingPBar` context manager handles the creation and updating of the shared progress bar processes. The following is the recommended way to create progress bars using multiprocessing:

```python
from langchain_progress import MultiprocessingPBarManager

def process_shard(shard, pbar):
    embeddings = HuggingFaceEmbeddings(model_name='sentence-transformers/all-MiniLM-L6-v2')

    with ProgressManager(embeddings, pbar):
        result = FAISS.from_documents(shard, embeddings)

    return result

doc_shards = np.array_split(docs, num_shards)

with MultiprocessingPBar(total=len(docs)) as pbar, Pool(num_shards) as pool:
    vectors = pool.starmap(process_shard, [(shard, pbar) for shard in doc_shards])
```

A full example can be found in `./examples/multiprocessing_example.py`.

## Tests

To run the test suite, you can run the following command from the root directory. Tests will be skipped if the required optional libraries are not installed:

```bash
python -m unittest
```

## Limitations

This wrapper cannot create progress bars for any API based embedding tool such as `HuggingFaceInferenceAPIEmbeddings` as it relies on wrapping the texts supplied to the embeddings method. This obviously can't be done when querying a remote API. This module also doesn't currently support all of langchain's embedding classes. If your embedding class isn't yet supported, please open an issue and I'll take a look when I get time.
