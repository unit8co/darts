# Running experiments to compare models with Darts: 

**The code in this folder is still under development, it can contain bugs and is not yet fully documented.** 

This folder contains the following files:
* README.txt - This file
* experiments_script.py - A script containing the code to run the experiments. It allows to configure
    the experiments, create the necessary log files and run the experiments. 
* builders.py - A script containing the functions to build the different models.
* experiments_notebook.ipynb - The same content as experiments_script.py, but in a Jupyter notebook 
  plus some plotting code. 
* results_plots.ipynb - A Jupyter notebook containing the code to plot the results of the experiments.

## Configuring and running an experiment : 
The experiments_script.py script can be run from the command line with multiple arguments to configure 
the experiment. 
* To get help on the arguments : 
    ``python experiments_script.py -h``
* The main arguments to provide are: 
  * dataset string, as one of:
      * "ETTh1": ``ETTh1Dataset``,
      * "ETTh2": ``ETTh2Dataset``,
      * "ETTm1": ``ETTm1Dataset``,
      * "ETTm2": ``ETTm2Dataset``,
      * "ILINet": ``ILINetDataset``,
      * "Electricity": ``ElectricityDataset``
  
  * model string, as one of:
      * "TCN": ``TCNModel``,
      * "DLinear": ``DLinearModel``,
      * "NLinear": ``NLinearModel``,
      * "NHiTS": ``NHiTSModel``,
      * "LinearRegression": ``LinearRegressionModel``,
      * "lgbm": ``LightGBMModel``,
      * "xgb": ``XGBModel``
* The argument ``time_budget`` allows to set the time budget in seconds for the hyperparameters tuning.
* By default, the hyperparameters of the models are tuned by minimizing an objective based on a metric function 
evaluated on the validation set. The default metric is ``smape``, it can be changed with the ``--eval_metric`` argument.
* The script will fit and predict with the best model 5 times in order to get an estimation of the variability 
 (in standard deviation) of the accuracy metric on the test set as well as the training and inference time.

The models hyperparameters tuning is done using [ray tune](https://docs.ray.io/en/master/tune/index.html).
[Optuna](https://optuna.org/) is used as the default method to setup the hyperparameters' search space 
and its default TPESampler is used as the default sampler from this space. 

## Adding a new model: 
In order to add a new model, you need to add a new model builder in the builders.py file. Furthermore, a function to 
generate the parameters of this model needs to be added to experiments_script.py. 

## Example: 
To run the experiments for the TCN model on the ETTh1 dataset with a time budget of 10 minutes:
``python experiments_script.py --dataset ETTh1 --model TCN --time_budget 600``

