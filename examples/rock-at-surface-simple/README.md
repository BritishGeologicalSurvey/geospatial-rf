# README: Example implementation of rf-random-forest 

* Deomonstration of the geospatial random forest code using an annonymiszed terrain derivative dataset
* A model is initially trained and then tested
* Model then saved and available for inference
* Inference then demonstrated for now using the same dataset

* Set up environment - see [building_your_environment_WINDOWS.md](../../building_your_environment_WINDOWS.md)
* Test data:
	* [sample_derivatives.csv](./example_data/sample_derivatives.csv)
	* [sample_derivatives_for_inference.csv](./example_data/sample_derivatives_for_inference.csv)
* Model training notebook:
	* [01_rf-train-and-test.ipynb](./01_rf-train-and-test.ipynb)
* Model inference notebook:
	* [02_infer_new_data.ipynb](02_infer_new_data.ipynb)

## Details on the test data and this example

* Provided is a test dataset consisting of terrain derivatives across a square spatial extent
	- these were prior calculated from a digital elevation model
* In addition to the x,y position and the derivative values themselves, is a column called `rock_presence`
	- 1 means that a exposed bedrock was observed at that location
	- 0 means that a exposed bedrock was not observed at that location
* The model example here is being trained to predict rock presence/absence based on the `rock_presence` column, considering the derivatives provided
* For this example, an 80:20 test:train split is used (refer to the comments in the notebook)


## Assessing model performance << remove?

- Once model is trained, you can now test it using your test dataset
- Using the h2o package, this can be calculated using your trained model ("rf" - the h2o trained model object) and your test dataset ("hf_train" in the notebook):

```python
performance = rf.model_performance(test_data=hf_test)
print(performance)
```
- This gives a range of outputs of which the following are important to report:
	+ Accuracy 
	+ Precision
	+ Recall (or *sensitivity*)
	+ F1 
- Refer to info here: https://docs.h2o.ai/h2o/latest-stable/h2o-docs/performance-and-prediction.html
- In addition to these stats, a confusion matrix is also useful, showing the relative spread of True and False negative and positive predictions

- Using the Swantston data to train a model then:
	- using a 75/25 train/test split
	- upsampling of training data due to 1/0 value imbalance (see https://scikit-learn.org/stable/modules/generated/sklearn.utils.resample.html)
		- generates syntheic points in the training data but not in the test data
	- model was not set to exclude contiguous data
	- MSE: 0.016290688448274887
	- RMSE: 0.12763498128755646
	- LogLoss: 0.06719542010059848
	- Mean Per-Class Error: 0.30120274914089346
	- AUC: 0.9202691867124856
	- AUCPR: 0.4006750752270021
	- Gini: 0.8405383734249712 - see https://docs.h2o.ai/h2o/latest-stable/h2o-docs/performance-and-prediction.html#gini-coefficient
	- Key values (here based on maximum metrics - not all that useful - see more specific values below)
		- Model precision:  [[0.9893877030232037 (threshold), 1.0 (value) ]]
		- Model recall (sensitivity):  [[0.008407787402478167 (threshold), 1.0 (value)]]
		- Model F1:  [[0.9331553115435355 (threshold), 0.4897959183673469 (value)]]
	- Key values according to classification targets of 0/1 (i.e. not rock/rock)

    | | precision| recall | f1-score | support | 
    |---|---|---|---|---|
    | 0 |      0.99  |    0.99 |     0.99  |    2910|
    | 1|       0.44 |     0.40|      0.42 |       30|
    |    accuracy|       | |                   0.99 |     2940|
    |   macro avg|       0.72   |   0.70 |     0.71   |   2940|
    |weighted avg|       0.99    |  0.99|      0.99    |  2940|

	- Confusion Matrix (Act/Pred) for max f1 @ threshold = 0.9331553115435355     	

    | |0| 1| Error| Rate|
    |---|---|---|---|---|
    |0 | 2903|7 |0.0024| (7.0/2910.0) |
    |1 | 18 |12 |0.6 | (18.0/30.0) |
    |Total | 2921 | 19 | 0.0085 |(25.0/2940.0)|

