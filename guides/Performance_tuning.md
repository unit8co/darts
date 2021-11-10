# Darts Performance Tuning Guide

This guide briefly recaps the main factors impact the performance when
training and using models. It mainly targets deep learning models in Darts.

## Main Recommendations

### Build your `TimeSeries` using 32-bits data
The models in Darts will dynamically cast themselves (to 64 or 32-bits)
to follow the dtype in the `TimeSeries`. Large performance and memory gains
can often be obtained when everything (data and model) is in float32.
To achieve this, all you have to do is make sure that the dtype of the
array (or Dataframe-backing array) is of dtype `np.float32`.

### Use a GPU
In many cases using a GPU will provide a drastic speedup compared to CPU.
It can also incur some overheads (for transferring data to/from the GPU),
so some testing and tuning is often necessary.

### Play with the batch size
Larger batch sizes tend to speed up the training because it reduces the number
of backward passes per epoch and has the potential to better parallelize computation. However it also changes the training dynamics (e.g. you might need more epochs, and the convergence dynamics is affected). Furthermore larger batch sizes increase memory consumption.So here too some testing is required.

### Play with `num_loader_workers`
All deep learning models in Darts have a parameter `num_loader_workers` which
configures the `num_workers` parameter in the PyTorch `DataLoaders`. By default
it is set to 0, which means that the main process will also take care of loading the data. Setting `num_workers > 0` will use additional processes to load the data. This typically incurs some overhead, but in many cases it can also substantially improve performance. The ideal value depends on many factors such as the batch size, whether you are using a GPU, and other things.

### Play with model size
Of course one of the main factor affecting performance is the model size
(number of parameters) and the number of operations required by forward/backward passes. Models in Darts can be tuned (e.g. number of layers, attention heads, widths etc), and these hyper-parameters tend to have a large impact on performance. When starting out, it is a good idea to build models of modest size first.

### Data in Memory and I/O bottlenecks
It's helpful to load all your `TimeSeries` in memory upfront if you can.
Darts offers the possibility to train models on any `Sequence[TimeSeries]`,
which means that for big datasets, you can write your own `Sequence` implementation, and read data lazily from disk. This will typically incur a high I/O cost, though. So when training on multiple series, first try to build a simple `List[TimeSeries]` upfront, and see if it holds in the computer memory.