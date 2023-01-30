* time_budget used : 1800 secodns (30 minutes)
* lightgbm gives the same accuracy metric over all 5 runs, potentially
because the seed is fixed and the same training data is used over the 
5 runs. 
* The components in this dataset vary widely. Some have a trend, all of
them have some seasonal pattern (at different scales). 
* Training and inference times are influenced by the overall load on the 
machine. For example, running multiple experiments at the same time
will delay all of them. 

The following commands were used to run the experiments:

```bash
python experiments_script.py --dataset ETTh1 --model NHiTS --time_budget 1800
python experiments_script.py --dataset ETTh1 --model TCN --time_budget 1800
python experiments_script.py --dataset ETTh1 --model lgbm --time_budget 1800
python experiments_script.py --dataset ETTh1 --model LinearRegression --time_budget 1800
python experiments_script.py --dataset ETTh1 --model DLinear --time_budget 1800
python experiments_script.py --dataset ETTh1 --model NLinear --time_budget 1800
```
The rest of the arguments were left to the default values. 

* NHiTS model checkpoints are too large to commit to the repo.