import pandas as pd
from dotenv import load_dotenv
import os
import tempfile
from typing import Optional
from darts import TimeSeries
from darts.models.forecasting.forecasting_model import ForecastingModel
from huggingface_hub import snapshot_download, upload_folder, create_repo


class HFHub:
    """
    HuggingFace Hub integration using official HF API.
    https://huggingface.co/docs/huggingface_hub/v0.20.3/en/guides/integrations
    """

    def __init__(self, api_key: Optional[str] = None):
        if api_key is None:
            # load from .env file or OS vars if available
            load_dotenv(override=True)
            api_key = os.getenv("HF_TOKEN")
            assert (
                api_key is not None
            ), "Could not find HF_TOKEN in OS environment. Cannot interact with HF Hub."
        self.HF_TOKEN = api_key

    def upload_model(
        self,
        repo_id: str = None,
        model: ForecastingModel = None,
        private: Optional[bool] = True,
    ):
        # Create repo if not existing yet and get the associated repo_id
        create_repo(repo_id=repo_id, repo_type="model", private=private, exist_ok=True)

        with tempfile.TemporaryDirectory() as tmpdirname:
            # print("created temporary directory for model", tmpdirname)
            model.save(path=f"{tmpdirname}/{model.model_name}")
            upload_folder(repo_id=repo_id, folder_path=tmpdirname, token=self.HF_TOKEN)

    def download_model(
        self,
        repo_id: str = None,
        model_name: str = None,
        model_class: object = None,
    ) -> ForecastingModel:
        with tempfile.TemporaryDirectory() as tmpdirname:
            snapshot_download(
                repo_id=repo_id, local_dir=tmpdirname, token=self.HF_TOKEN
            )
            model = model_class.load(path=f"{tmpdirname}/{model_name}")
            return model

    def upload_timeseries(
        self,
        repo_id: str = None,
        series: TimeSeries = None,
        series_name: str = None,
        private: Optional[bool] = True,
    ):
        # Create repo if not existing yet and get the associated repo_id
        repo_info = create_repo(
            repo_id=repo_id, repo_type="dataset", private=private, exist_ok=True
        )
        # print(f"repo_info: ", repo_info)
        df = series.pd_dataframe()
        with tempfile.TemporaryDirectory() as tmpdirname:
            df.to_parquet(path=f"{tmpdirname}/{series_name}.parquet")
            upload_folder(
                repo_id=repo_id,
                repo_type="dataset",
                folder_path=tmpdirname,
                token=self.HF_TOKEN,
            )

    def download_timeseries(
        self,
        repo_id: str = None,
        series_name: str = None,
    ) -> TimeSeries:
        with tempfile.TemporaryDirectory() as tmpdirname:
            snapshot_download(
                repo_id=repo_id,
                repo_type="dataset",
                local_dir=tmpdirname,
                token=self.HF_TOKEN,
            )
            print(os.listdir(tmpdirname))
            df = pd.read_parquet(
                f"{tmpdirname}/{series_name}.parquet", engine="pyarrow"
            )
            ts = TimeSeries.from_dataframe(df)
            return ts
