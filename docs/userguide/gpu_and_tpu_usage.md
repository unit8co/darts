# Using Torch Models with GPUs and TPUs
This section was written for Darts 0.17.0 and later.

We assume that you already know about Torch Forecasting Models in Darts. If you're new to the topic we recommend you to read the [guide on Torch Forecasting Models](https://unit8co.github.io/darts/userguide/torch_forecasting_models.html) first. This guide also contains a section about performance recommendations, which we recommend reading first. Finally, here is also an [Recurrent Neural Network (RNN) Model example](https://unit8co.github.io/darts/examples/04-RNN-examples.html), on which this section is going to be based on.

## Use CPU

By default all models will run on CPU. As shown in the RNN example above, we'll import the Air Passenger dataset, as well as other necessary modules.
```python
import numpy as np
import pandas as pd

import matplotlib.pyplot as plt

from darts.dataprocessing.transformers import Scaler
from darts.models import RNNModel
from darts.metrics import mape
from darts.datasets import AirPassengersDataset
```

Now we read and scale the data like this:

```python
# Read data:
series = AirPassengersDataset().load()
series = series.astype(np.float32)

# Create training and validation sets:
train, val = series.split_after(pd.Timestamp("19590101"))

# Normalize the time series (note: we avoid fitting the transformer on the validation set)
transformer = Scaler()
train_transformed = transformer.fit_transform(train)
val_transformed = transformer.transform(val)
series_transformed = transformer.transform(series)
```

Next we will create our RNN like this:
```python
my_model = RNNModel(
    model="RNN",
    hidden_dim=20,
    dropout=0,
    batch_size=16,
    n_epochs=300,
    optimizer_kwargs={"lr": 1e-3},
    model_name="Air_RNN",
    log_tensorboard=True,
    random_state=42,
    training_length=20,
    input_chunk_length=14,
    force_reset=True,
)
```

and fit it to the data:
```python
my_model.fit(train_transformed, val_series=val_transformed)
```

where in the output we can see that no other processing unit is used to train our model:
```
GPU available: False, used: False
TPU available: False, using: 0 TPU cores
IPU available: False, using: 0 IPUs

  | Name      | Type    | Params
--------------------------------------
0 | criterion | MSELoss | 0     
1 | rnn       | RNN     | 460   
2 | V         | Linear  | 21    
--------------------------------------
481       Trainable params
0         Non-trainable params
481       Total params
0.004     Total estimated model params size (MB)

Epoch 299: 100% 8/8 [00:00<00:00, 42.49it/s, loss=0.00285, v_num=logs]
<darts.models.forecasting.rnn_model.RNNModel at 0x7fc75901cb10>
```

Now the model is ready to start predicting, which won't be shown here since it's included in the example linked in the start of this guide.

## Use a GPU
GPUs can dramatically improve the performance of your model in terms of processing time. By using an Accelerator in the [Pytorch Lightning Trainer](https://pytorch-lightning.readthedocs.io/en/stable/common/trainer.html#accelerator), we can enjoy the benefits of a GPU. We only need to instruct our model to use our machine's GPU through PyTorch Lightning Trainer parameters, which are expressed as the `pl_trainer_kwargs` dictionary, like this:
```python
my_model = RNNModel(
    model="RNN",
    ...
    force_reset=True,
    pl_trainer_kwargs={
      "accelerator": "gpu",
      "gpus": [0]
    },
)
```

which now outputs:
```
GPU available: True, used: True
TPU available: False, using: 0 TPU cores
IPU available: False, using: 0 IPUs
LOCAL_RANK: 0 - CUDA_VISIBLE_DEVICES: [0]

  | Name      | Type    | Params
--------------------------------------
0 | criterion | MSELoss | 0     
1 | rnn       | RNN     | 460   
2 | V         | Linear  | 21    
--------------------------------------
481       Trainable params
0         Non-trainable params
481       Total params
0.004     Total estimated model params size (MB)

Epoch 299: 100% 8/8 [00:00<00:00, 39.81it/s, loss=0.00285, v_num=logs]
<darts.models.forecasting.rnn_model.RNNModel at 0x7ff1b5e4d4d0>
```

From the output we can see that the GPU is both available and used. The rest of the code doesn't require any change, i.e. it's irrelevant if we are using a GPU or CPU.

## Use a TPU

Tensor Processing Unit (TPU) is an AI accelerator application-specific integrated circuit (ASIC) developed by Google specifically for neural network machine learning.

There are three main ways to get access to a TPU:

* Google Colab
* Google Cloud (GCP)
* Kaggle

If you are using a TPU in the Google Colab kind of notebook, then you should first install these:
```
!pip install cloud-tpu-client==0.10 https://storage.googleapis.com/tpu-pytorch/wheels/torch_xla-1.9-cp37-cp37m-linux_x86_64.whl
!pip install torch==1.9.0+cu111 torchvision==0.10.0+cu111 torchtext==0.10.0 -f https://download.pytorch.org/whl/cu111/torch_stable.html
!pip install pyyaml==5.4.1
```

and then instruct our model to use a TPU or more. In our example we are using four TPUs, like this:
```python
my_model = RNNModel(
    model="RNN",
    ...
    force_reset=True,
    pl_trainer_kwargs={
      "accelerator": "tpu",
      "tpu_cores": [4]
    },
)
```

which outputs:
```
WARNING:root:TPU has started up successfully with version pytorch-1.9
GPU available: False, used: False
TPU available: True, using: [4] TPU cores
IPU available: False, using: 0 IPUs

  | Name      | Type    | Params
--------------------------------------
0 | criterion | MSELoss | 0     
1 | rnn       | RNN     | 460   
2 | V         | Linear  | 21    
--------------------------------------
481       Trainable params
0         Non-trainable params
481       Total params
0.002     Total estimated model params size (MB)
Epoch 299: 100% 8/8 [00:00<00:00, 8.52it/s, loss=0.00285, v_num=logs]
<darts.models.forecasting.rnn_model.RNNModel at 0x7ff1b5e4d4d0>
```

From the output we can see that our model is using 4 TPUs.