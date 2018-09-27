from .timeseries_model import SupervisedTimeSeriesModel
from sklearn.ensemble import RandomForestRegressor


class SupervisedRegression(SupervisedTimeSeriesModel):
    # TODO: Not really needed at this point

    def __init__(self, model=RandomForestRegressor(n_estimators=100, n_jobs=-1, random_state=0)):
        """
        :param model: The regression model to use. It can be any sklearn model with fit() and predict()
        """
        super(SupervisedRegression, self).__init__()
        self.model = model

    def __str__(self):
        return 'supervised ({})'.format(self.model)

    def fit(self, df, target_column, feature_columns=None):

        # TODO: encode feature columns

        if feature_columns is None:
            Xtrain = df.drop([target_column], axis=1)
        else:
            Xtrain = df[feature_columns]

        ytrain = df[target_column].values
        self.model.fit(Xtrain, ytrain)

    def predict(self, Xtest):
        return self.model.predict(Xtest)
