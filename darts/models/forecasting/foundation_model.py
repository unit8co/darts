"""
Time Series Foundation Model (TSFM)
-----------------------------------

This file contains several abstract classes:

    * FoundationModel: base class for foundation forecasting models with PyTorch Lightning backend,
        inheriting from :class:`MixedCovariatesTorchModel` and :class:`TorchForecastingModel`.
"""

from abc import ABC

from darts.logging import get_logger
from darts.models.forecasting.torch_forecasting_model import MixedCovariatesTorchModel

logger = get_logger(__name__)


class FoundationModel(MixedCovariatesTorchModel, ABC):
    def __init__(self, **kwargs):
        """Foundation Forecasting Model with PyTorch Lightning backend.

        This class is meant to be inherited to create a new foundation forecasting model.
        It governs the interactions between:
        - Darts forecasting models (module) :class:`PLTorchForecastingModel`
        - Darts integrated PL Lightning Trainer :class:`pytorch_lightning.Trainer` or custom PL Trainers
        - Dataset loaders :class:`TorchTrainingDataset` and :class:`TorchInferenceDataset` or custom Dataset
          Loaders.

        This class itself inherits from :class:`MixedCovariatesTorchModel`, which in turn inherits from
        :class:`TorchForecastingModel`. That allows :class:`FoundationModel` to use functionalities from both,
        such as optimized historical forecasting, model training (fine-tuning), checkpointing, and more.

        When subclassing this class, please make sure to perform necessary parameter validation and then call
        super().__init__(**kwargs). Also, please implement the abstract method :func:`_create_model()`.

        If the model requires downloading configuration files and model weights from HuggingFace, please
        instantiate a :class:`HuggingFaceConnector` and use its methods to load the model configuration
        inside :func:`__init__()` and to load the model weights inside :func:`_create_model()`.


        .. tip::
            You can perform full or partial fine-tuning of the model by setting the ``enable_finetuning`` parameter.
            Read more in the parameter description below and in the `Fine-Tuning Examples
            <https://unit8co.github.io/darts/examples/27-Torch-and-Foundation-Model-Fine-Tuning-examples.html>`__.

        Parameters
        ----------
        batch_size
            Number of time series (input and output sequences) used in each fine-tuning pass. Default: ``32``.
        n_epochs
            Number of epochs over which to fine-tune the model. Default: ``100``.
        model_name
            Name of the model. Used for creating checkpoints and saving tensorboard data. If not specified,
            defaults to the following string ``"YYYY-mm-dd_HH_MM_SS_torch_model_run_PID"``, where the initial part
            of the name is formatted with the local date and time, while PID is the processed ID (preventing models
            spawned at the same time by different processes to share the same model_name). E.g.,
            ``"2021-06-14_09_53_32_torch_model_run_44607"``.
        work_dir
            Path of the working directory, where to save checkpoints and Tensorboard summaries.
            Default: current working directory.
        log_tensorboard
            If set, use Tensorboard to log the different parameters. The logs will be located in:
            ``"{work_dir}/darts_logs/{model_name}/logs/"``. Default: ``False``.
        nr_epochs_val_period
            Number of epochs to wait before evaluating the validation loss (if a validation
            ``TimeSeries`` is passed to the :func:`fit()` method). Default: ``1``.
        force_reset
            If set to ``True``, any previously-existing model with the same name will be reset (all checkpoints will
            be discarded). Default: ``False``.
        save_checkpoints
            Whether to automatically save the untrained model and checkpoints from training.
            To load the model from checkpoint, call :func:`MyModelClass.load_from_checkpoint()`, where
            :class:`MyModelClass` is the :class:`TorchForecastingModel` class that was used (such as :class:`TFTModel`,
            :class:`NBEATSModel`, etc.). If set to ``False``, the model can still be manually saved using
            :func:`save()` and loaded using :func:`load()`. Default: ``False``.
        add_encoders
            A large number of past and future covariates can be automatically generated with `add_encoders`.
            This can be done by adding multiple pre-defined index encoders and/or custom user-made functions that
            will be used as index encoders. Additionally, a transformer such as Darts' :class:`Scaler` can be added to
            transform the generated covariates. This happens all under one hood and only needs to be specified at
            model creation.
            Read :meth:`SequentialEncoder <darts.dataprocessing.encoders.SequentialEncoder>` to find out more about
            ``add_encoders``. Default: ``None``. An example showing some of ``add_encoders`` features:

            .. highlight:: python
            .. code-block:: python

                def encode_year(idx):
                    return (idx.year - 1950) / 50

                add_encoders={
                    'cyclic': {'future': ['month']},
                    'datetime_attribute': {'future': ['hour', 'dayofweek']},
                    'position': {'past': ['relative'], 'future': ['relative']},
                    'custom': {'past': [encode_year]},
                    'transformer': Scaler(),
                    'tz': 'CET'
                }
            ..
        random_state
            Controls the randomness of the weights initialization and reproducible forecasting.
        pl_trainer_kwargs
            By default :class:`TorchForecastingModel` creates a PyTorch Lightning Trainer with several useful presets
            that performs the training, validation and prediction processes. These presets include automatic
            checkpointing, tensorboard logging, setting the torch device and more.
            With ``pl_trainer_kwargs`` you can add additional kwargs to instantiate the PyTorch Lightning trainer
            object. Check the `PL Trainer documentation
            <https://pytorch-lightning.readthedocs.io/en/stable/common/trainer.html>`__ for more information about the
            supported kwargs. Default: ``None``.
            Running on GPU(s) is also possible using ``pl_trainer_kwargs`` by specifying keys ``"accelerator",
            "devices", and "auto_select_gpus"``. Some examples for setting the devices inside the ``pl_trainer_kwargs``
            dict:

            - ``{"accelerator": "cpu"}`` for CPU,
            - ``{"accelerator": "gpu", "devices": [i]}`` to use only GPU ``i`` (``i`` must be an integer),
            - ``{"accelerator": "gpu", "devices": -1, "auto_select_gpus": True}`` to use all available GPUS.

            For more info, see here:
            `trainer flags
            <https://pytorch-lightning.readthedocs.io/en/stable/common/trainer.html#trainer-flags>`__,
            and `training on multiple gpus
            <https://pytorch-lightning.readthedocs.io/en/stable/accelerators/gpu_basic.html#train-on-multiple-gpus>`__.

            With parameter ``"callbacks"`` you can add custom or PyTorch-Lightning built-in callbacks to Darts'
            :class:`TorchForecastingModel`. Below is an example for adding EarlyStopping to the training process.
            The model will stop training early if the validation loss `val_loss` does not improve beyond
            specifications. For more information on callbacks, visit:
            `PyTorch Lightning Callbacks
            <https://pytorch-lightning.readthedocs.io/en/stable/extensions/callbacks.html>`__

            .. highlight:: python
            .. code-block:: python

                from pytorch_lightning.callbacks.early_stopping import EarlyStopping

                # stop training when validation loss does not decrease more than 0.05 (`min_delta`) over
                # a period of 5 epochs (`patience`)
                my_stopper = EarlyStopping(
                    monitor="val_loss",
                    patience=5,
                    min_delta=0.05,
                    mode='min',
                )

                pl_trainer_kwargs={"callbacks": [my_stopper]}
            ..

            Note that you can also use a custom PyTorch Lightning Trainer for training and prediction with optional
            parameter ``trainer`` in :func:`fit()` and :func:`predict()`.
        show_warnings
            whether to show warnings raised from PyTorch Lightning. Useful to detect potential issues of
            your forecasting use case. Default: ``False``.
        enable_finetuning
            Enables model fine-tuning. Only effective if not ``None``.
            If a bool, specifies whether to perform full fine-tuning / training (all parameters are updated) or keep
            all parameters frozen. If a dict, specifies which parameters to fine-tune. Must only contain one key-value
            record. Can be used to:

            - Unfreeze specific parameters, while keeping everything else frozen:
              ``{"unfreeze": ["param.name.patterns.*"]}``
            - Freeze specific parameters, while keeping everything else unfrozen:
              ``{"freeze": ["param.name.patterns.*"]}``

            Default: ``None``.
        """
        # Set default fine-tuning to False for foundation models
        if "enable_finetuning" not in self.model_params:
            self.model_params["enable_finetuning"] = False

        # initialize `TorchForecastingModel` base class
        super().__init__(**self._extract_torch_model_params(**self.model_params))

        # extract pytorch lightning module kwargs
        self.pl_module_params = self._extract_pl_module_params(**self.model_params)

        # pass fine-tuning flag to the PLModule so it can set up training-specific
        # quantile handling (separate from prediction-time likelihood)
        if self.enable_finetuning:
            self.pl_module_params["enable_finetuning"] = True

        use_reversible_instance_norm: bool | dict = self.pl_module_params.get(
            "use_reversible_instance_norm", False
        )
        if use_reversible_instance_norm is True or (
            isinstance(use_reversible_instance_norm, dict)
            and use_reversible_instance_norm.get("affine", True)
        ):
            if use_reversible_instance_norm is True:
                use_reversible_instance_norm = dict(affine=False)
            else:
                use_reversible_instance_norm["affine"] = False
            logger.warning(
                f"By default, Reversible Instance Normalization (RINorm) in Darts inserts affine transformation "
                f"weights, which do not exist in foundation model checkpoints. To prevent incompatible model "
                f"weights when loading checkpoints, `use_reversible_instance_norm` is overridden to "
                f"`{use_reversible_instance_norm}`."
            )
            self.pl_module_params["use_reversible_instance_norm"] = (
                use_reversible_instance_norm
            )
