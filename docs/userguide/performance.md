# Neural Networks Performance Recommendations

This document recaps the main factors impacting the performance when
training and using models. It mainly targets deep learning models in Darts.

## Build your `TimeSeries` using 32-bits data
The models in Darts will dynamically cast themselves (to 64 or 32-bits)
to follow the dtype in the `TimeSeries`. Large performance and memory gains
can often be obtained when everything (data and model) is in float32.
To achieve this, it is enough to build your `TimeSeries` from arrays (or Dataframe-backing array) having dtype `np.float32`, or simply call `my_series32 = my_series.astype(np.float32)`. Calling `my_series.dtype` gives you the dtype of your `TimeSeries`.

## Use a GPU
In many cases using a GPU will provide a drastic speedup compared to CPU.
It can also incur some overheads (for transferring data to/from the GPU),
so some testing and tuning is often necessary. If a CUDA-enabled GPU is present on your
system, Darts will attempt to use it by default. You can specify
`torch_device_str` (giving a string such as `"cuda"` or `"cpu"`) to control this.

## Play with the batch size
A larger batch size tends to speed up the training because it reduces the number
of backward passes per epoch and has the potential to better parallelize computation. However it also changes the training dynamics (e.g. you might need more epochs, and the convergence dynamics is affected). Furthermore larger batch sizes increase memory consumption. So here too some testing is required.

## Play with `num_loader_workers`
All deep learning models in Darts have a parameter `num_loader_workers` which
configures the `num_workers` parameter in the PyTorch `DataLoaders`. By default
it is set to 0, which means that the main process will also take care of loading the data. Setting `num_workers > 0` will use additional workers to load the data. This typically incurs some overhead (notably increasing memory consumption), but in some cases it can also substantially improve performance. The ideal value depends on many factors such as the batch size, whether you are using a GPU and the number of CPU cores available.

## Small models first
Of course one of the main factors affecting performance is the model size
(number of parameters) and the number of operations required by forward/backward passes. Models in Darts can be tuned (e.g. number of layers, attention heads, widths etc), and these hyper-parameters tend to have a large impact on performance. When starting out, it is a good idea to build models of modest size first.

## Data in Memory and I/O bottlenecks
It's helpful to load all your `TimeSeries` in memory upfront if you can.
Darts offers the possibility to train models on any `Sequence[TimeSeries]`,
which means that for big datasets, you can write your own `Sequence` implementation, and read the time series lazily from disk. This will typically incur a high I/O cost, though. So when training on multiple series, first try to build a simple `List[TimeSeries]` upfront, and see if it holds in the computer memory.

## Do not use *all* possible sub-series for training
By default, when calling `fit()`, the models in Darts will build a `TrainingDataset` instance that is
suitable for the model that you are using (e.g., `PastCovariatesTorchModel`, `FutureCovariatesTorchModel`, etc.).
By default, these training datasets will often contain *all* possible consecutive (input, output) subseries present
in each `TimeSeries`. If your `TimeSeries` are long, this can result in a large amount of training samples, which directly (linearly)
impacts the time required to train the model for one epoch. You have two options to limit this:

* Specify some `max_samples_per_ts` argument to the `fit()` function. This will use only the most recent `max_samples_per_ts` samples
per `TimeSeries` for training.
* If this option does not do what you want, you can implement your own `TrainingDataset` instance, and define
how to slice your `TimeSeries` for training yourself. We suggest to have a look at [this submodule](https://github.com/unit8co/darts/tree/master/darts/utils/data)
to see examples of how to do it.


-------------

## Example Benchmark
As an example, we show here the time required to train one epoch on the first 80% of the energy dataset (`darts.datasets.EnergyDataset`), which consists of one multivariate series that is 28050 timesteps long and has 28 dimensions.
We train two models; `NBEATSModel` and `TFTModel`, with default parameters and `input_chunk_length=48` and `output_chunk_length=12` (which results in 27991 training samples with default sequential training datasets). For the TFT model, we also set the parameter `add_cyclic_encoder='hour'`. The tests are made on a Intel CPU i9-10900K CPU @ 3.70GHz, with an Nvidia RTX 2080s GPU, 32 GB of RAM. All `TimeSeries` are pre-loaded in memory and given to the models as a list.

| Model         | Dataset| dtype | CUDA | Batch size | num workers  | time per epoch |
| ------------- | ------ | ---- | ---- | ---------- | ------------ | -------------- |
| `NBEATSModel` | Energy | 64   | no   | 32         | 0            | 283s           |
| `NBEATSModel` | Energy | 64   | no   | 32         | 2            | 285s           |
| `NBEATSModel` | Energy | 64   | no   | 32         | 4            | 282s           |
| `NBEATSModel` | Energy | 64   | no   | 1024       | 0            | 58s            |
| `NBEATSModel` | Energy | 64   | no   | 1024       | 2            | 57s            |
| `NBEATSModel` | Energy | 64   | no   | 1024       | 4            | 58s            |
| `NBEATSModel` | Energy | 64   | yes  | 32         | 0            | 63s            |
| `NBEATSModel` | Energy | 64   | yes  | 32         | 2            | 62s            |
| `NBEATSModel` | Energy | 64   | yes  | 1024       | 0            | 13.3s          |
| `NBEATSModel` | Energy | 64   | yes  | 1024       | 2            | 12.1s          |
| `NBEATSModel` | Energy | 64   | yes  | 1024       | 4            | 12.3s          |
|               |                  |      |      |            |              |                |
| `NBEATSModel` | Energy | 32   | no   | 32         | 0            | 117s           |
| `NBEATSModel` | Energy | 32   | no   | 32         | 2            | 115s           |
| `NBEATSModel` | Energy | 32   | no   | 32         | 4            | 117s           |
| `NBEATSModel` | Energy | 32   | no   | 1024       | 0            | 28.4s          |
| `NBEATSModel` | Energy | 32   | no   | 1024       | 2            | 27.4s          |
| `NBEATSModel` | Energy | 32   | no   | 1024       | 4            | 27.5s          |
| `NBEATSModel` | Energy | 32   | yes  | 32         | 0            | 41.5s          |
| `NBEATSModel` | Energy | 32   | yes  | 32         | 2            | 40.6s          |
| `NBEATSModel` | Energy | 32   | yes  | 1024       | 0            | 2.8s           |
| `NBEATSModel` | Energy | 32   | yes  | 1024       | 2            | 1.65           |
| `NBEATSModel` | Energy | 32   | yes  | 1024       | 4            | 1.8s           |
|               |                  |      |      |            |              |                |
| `TFTModel`  | Energy | 64   | no   | 32         | 0            | 78s            |
| `TFTModel`  | Energy | 64   | no   | 32         | 2            | 72s            |
| `TFTModel`  | Energy | 64   | no   | 32         | 4            | 72s            |
| `TFTModel`  | Energy | 64   | no   | 1024       | 0            | 46s            |
| `TFTModel`  | Energy | 64   | no   | 1024       | 2            | 38s            |
| `TFTModel`  | Energy | 64   | no   | 1024       | 4            | 39s            |
| `TFTModel`  | Energy | 64   | yes  | 32         | 0            | 125s           |
| `TFTModel`  | Energy | 64   | yes  | 32         | 2            | 115s           |
| `TFTModel`  | Energy | 64   | yes  | 1024       | 0            | 59s            |
| `TFTModel`  | Energy | 64   | yes  | 1024       | 2            | 50s            |
| `TFTModel`  | Energy | 64   | yes  | 1024       | 4            | 50s            |
|               |                  |      |      |            |              |                |
| `TFTModel`  | Energy | 32   | no   | 32         | 0            | 70s            |
| `TFTModel`  | Energy | 32   | no   | 32         | 2            | 62.6s          |
| `TFTModel`  | Energy | 32   | no   | 32         | 4            | 63.6           |
| `TFTModel`  | Energy | 32   | no   | 1024       | 0            | 31.9s          |
| `TFTModel`  | Energy | 32   | no   | 1024       | 2            | 45s            |
| `TFTModel`  | Energy | 32   | no   | 1024       | 4            | 44s            |
| `TFTModel`  | Energy | 32   | yes  | 32         | 0            | 73s            |
| `TFTModel`  | Energy | 32   | yes  | 32         | 2            | 58s            |
| `TFTModel`  | Energy | 32   | yes  | 1024       | 0            | 41s            |
| `TFTModel`  | Energy | 32   | yes  | 1024       | 2            | 31s            |
| `TFTModel`  | Energy | 32   | yes  | 1024       | 4            | 31s            |
