# prevent PyTorch Lightning from using GPU (M1 system compatibility)
tfm_kwargs = {"pl_trainer_kwargs": {"accelerator": "cpu"}}
