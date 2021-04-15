import numpy as np
import pandas as pd

from ..logging import get_logger
from ..timeseries import TimeSeries
from .base_test_class import DartsBaseTestClass
from ..models.random_forest import RandomForest
logger = get_logger(__name__)


class RandomForestTestCase(DartsBaseTestClass):
    __test__ = True
    data_dict = {"Time": pd.date_range(start="20130501", end="20200301", freq="MS")}
    data_dict["Values1"] = np.random.uniform(low=-10, high=10, size=len(data_dict["Time"]))
    data_dict["Values2"] = np.random.uniform(low=0, high=1, size=len(data_dict["Time"]))
    data_dict["Exog1"] = np.random.uniform(low=-5, high=-4, size=len(data_dict["Time"]))
    data_dict["Exog2"] = np.random.uniform(low=0, high=5, size=len(data_dict["Time"]))

    data = TimeSeries.from_dataframe(df=pd.DataFrame(data_dict), time_col="Time")

    def test_creation(self):
        lgbm = RandomForest(lags=5)
        self.assertEqual(lgbm.lags, [1, 2, 3, 4, 5])

        lgbm = RandomForest(lags=5, lags_exog=3)
        self.assertEqual(lgbm.lags_exog, [1, 2, 3])

        lgbm = RandomForest(lags=5, lags_exog=True)
        self.assertEqual(lgbm.lags, [1, 2, 3, 4, 5])

        lgbm = RandomForest(lags=5, lags_exog=False)
        self.assertEqual(lgbm.lags_exog, [0])

        lgbm = RandomForest(lags=5, lags_exog=[3, 6, 9, 12])
        self.assertEqual(lgbm.lags_exog, [3, 6, 9, 12])

        with self.assertRaises(ValueError):
            RandomForest(lags=-5)
        with self.assertRaises(ValueError):
            RandomForest(lags=3.6)


    def test_create_lagged_data(self):
        nr_lags = 12
        lgbm = RandomForest(lags=nr_lags)

        lagged_data = lgbm._create_lagged_data(series=self.data[["Values1"]], lags=lgbm.lags, keep_current=True)
        self.assertEqual(len(lagged_data), len(self.data)-nr_lags)
        self.assertEqual(lagged_data.shape[1], nr_lags+1)
        self.assertTrue(
            np.all(np.array_equal(
                lagged_data.iloc[0, :],
                self.data.pd_dataframe()["Values1"][:nr_lags+1][::-1]
                )
            )
        )

        lags = [3, 6, 7]
        lgbm = RandomForest(lags=lags)
        lagged_data = lgbm._create_lagged_data(series=self.data[["Values1"]], lags=lgbm.lags, keep_current=True)
        self.assertEqual(
            lagged_data.columns.values.tolist(),
            ['Values1', 'Values1_lag3', 'Values1_lag6', 'Values1_lag7']
        )
        self.assertTrue(
            np.all(np.array_equal(
                lagged_data.iloc[0, :],
                self.data.pd_dataframe()["Values1"].iloc[[0, 1, 4, 7]][::-1]
                )
            )
        )

    def test_create_training_data(self):
        nr_lags = 12
        lgbm = RandomForest(lags=nr_lags, lags_exog=False)
        exog = ["Values2", "Exog1", "Exog2"]
        lagged_data_ex1 = lgbm._create_training_data(
            series=self.data[["Values1"]],
            exog=self.data[exog]
        ).pd_dataframe()
        self.assertEqual(len(lagged_data_ex1), len(self.data)-nr_lags)
        self.assertEqual(lagged_data_ex1.shape[1], nr_lags+1+len(exog))
        self.assertTrue(
            np.all(np.array_equal(
                lagged_data_ex1["Values1"].values,
                self.data.pd_dataframe()["Values1"].values[nr_lags:]
                )
            )
        )

    def test_fit(self):
        lgb1 = RandomForest(lags=12)
        lgb1.fit(series=self.data[["Values1"]])
        self.assertEqual(lgb1.nr_exog, 0)

        lgb2 = RandomForest(lags=12, lags_exog=True)
        lgb2.fit(series=self.data[["Values1"]], exog=self.data[["Values2", "Exog1", "Exog2"]])
        self.assertEqual(lgb2.nr_exog, 36)

        lgb2 = RandomForest(lags=12, lags_exog=False)
        lgb2.fit(series=self.data[["Values1"]], exog=self.data[["Values2", "Exog1", "Exog2"]])
        self.assertEqual(lgb2.nr_exog, 3)

        lgb2 = RandomForest(lags=12, lags_exog=[1, 4, 6])
        lgb2.fit(series=self.data[["Values1"]], exog=self.data[["Values2", "Exog1", "Exog2"]])
        self.assertEqual(lgb2.nr_exog, 9)

    def test_prediction(self):
        lgb1 = RandomForest(lags=12)
        lgb1.fit(series=self.data[["Values1"]])
        pred1 = lgb1.predict(n=12)
        self.assertEqual(len(pred1), 12)

        exog = self.data.pd_dataframe()[["Values2", "Exog1", "Exog2"]].iloc[:12, :]
        lgb2 = RandomForest(lags=12)
        lgb2.fit(series=self.data[["Values1"]], exog=self.data[["Values2", "Exog1", "Exog2"]])
        pred2 = lgb2.predict(n=12, exog=TimeSeries.from_dataframe(exog))
        self.assertEqual(len(pred2), 12)
        pred3 = lgb2.predict(n=12, exog=exog)
        self.assertEqual(len(pred3), 12)

    def test_performance(self):
        from darts.metrics import mape

        data = pd.read_csv('examples/ice_cream_heater.csv', delimiter=",")
        data1 = TimeSeries.from_dataframe(data[["Month", "heater"]], time_col="Month").diff(1)
        train, test = data1.split_before(pd.Timestamp("20180101"))
        lgb = RandomForest(lags=12, n_estimators=200)
        lgb.fit(series=train)
        pred = lgb.predict(n=len(test))
        print(mape(pred, test))

        # import matplotlib.pyplot as plt
        # train["heater"].plot()
        # test[:-1]["heater"].plot()
        # pred[1:].plot()
        # plt.show()

        data2 = TimeSeries.from_dataframe(data, time_col="Month").diff(1)
        train, test = data2.split_before(pd.Timestamp("20180101"))
        lgb = RandomForest(lags=12, n_estimators=200)
        lgb.fit(series=train["heater"], exog=train[["ice cream"]])
        pred = lgb.predict(n=len(test), exog=test[["ice cream"]])
        print(mape(pred, test["heater"]))

        # import matplotlib.pyplot as plt
        # train["heater"].plot()
        # test[:-1]["heater"].plot()
        # pred[1:].plot()
        # plt.show()