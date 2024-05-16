import sys

from pytorch_lightning.callbacks import TQDMProgressBar
from pytorch_lightning.callbacks.progress.tqdm_progress import Tqdm


class TFMProgressBar(TQDMProgressBar):
    def __init__(
        self,
        enable_sanity_check_bar: bool = True,
        enable_train_bar: bool = True,
        enable_validation_bar: bool = True,
        enable_prediction_bar: bool = True,
        enable_train_bar_only: bool = False,
        **kwargs,
    ):
        """Darts' Progress Bar for `TorchForecastingModels`.

        Allows to customize for which model stages (sanity checks, training, validation, prediction) to display a
        progress bar.

        This class is a PyTorch Lightning `Callback` and can be passed to the `TorchForecastingModel` constructor
        through the `pl_trainer_kwargs` parameter.

        Examples
        --------
        >>> from darts.models import NBEATSModel
        >>> from darts.utils.callbacks import TFMProgressBar
        >>> # only display the training bar and not the validation, prediction, and sanity check bars
        >>> prog_bar = TFMProgressBar(enable_train_bar_only=True)
        >>> model = NBEATSModel(1, 1, pl_trainer_kwargs={"callbacks": [prog_bar]})

        Parameters
        ----------
        enable_sanity_check_bar
            Whether to enable to progress bar for sanity checks.
        enable_train_bar
            Whether to enable to progress bar for training.
        enable_validation_bar
            Whether to enable to progress bar for validation.
        enable_prediction_bar
            Whether to enable to progress bar for prediction.
        enable_train_bar_only
            Whether to disable all progress bars except the bar for training.
        **kwargs
            Arguments passed to the PyTorch Lightning's `TQDMProgressBar
            <https://scikit-learn.org/stable/glossary.html#term-random_state>`_.
        """
        super().__init__(**kwargs)
        self.enable_sanity_check_bar = enable_sanity_check_bar
        self.enable_train_bar = enable_train_bar
        self.enable_validation_bar = enable_validation_bar
        self.enable_prediction_bar = enable_prediction_bar
        self.enable_train_bar_only = enable_train_bar_only

    def init_sanity_tqdm(self) -> Tqdm:
        """Override this to customize the tqdm bar for the validation sanity run."""
        return Tqdm(
            desc=self.sanity_check_description,
            position=(2 * self.process_position),
            disable=not self.enable_sanity_check_bar or self.enable_train_bar_only,
            leave=False,
            dynamic_ncols=True,
            file=sys.stdout,
        )

    def init_predict_tqdm(self) -> Tqdm:
        """Override this to customize the tqdm bar for predicting."""
        return Tqdm(
            desc=self.predict_description,
            position=(2 * self.process_position),
            disable=not self.enable_prediction_bar or self.enable_train_bar_only,
            leave=True,
            dynamic_ncols=True,
            file=sys.stdout,
            smoothing=0,
        )

    def init_train_tqdm(self) -> Tqdm:
        """Override this to customize the tqdm bar for training."""
        return Tqdm(
            desc=self.train_description,
            position=(2 * self.process_position),
            disable=not self.enable_train_bar,
            leave=True,
            dynamic_ncols=True,
            file=sys.stdout,
            smoothing=0,
        )

    def init_validation_tqdm(self) -> Tqdm:
        """Override this to customize the tqdm bar for validation."""
        # The train progress bar doesn't exist in `trainer.validate()`
        has_main_bar = self.trainer.state.fn != "validate"
        return Tqdm(
            desc=self.validation_description,
            position=(2 * self.process_position + has_main_bar),
            disable=not self.enable_validation_bar or self.enable_train_bar_only,
            leave=not has_main_bar,
            dynamic_ncols=True,
            file=sys.stdout,
        )
