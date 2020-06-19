## Regression about quality in Tensorflow 2
I created a simple regression model in Tensorflow 2 that can deal with the following datasets:  
* 1- wine quality dataset [link](https://www.kaggle.com/uciml/red-wine-quality-cortez-et-al-2009)   
* 2- real estate dataset  [link](https://data.world/datasets/real-estate)

## How to use
Training & Prediction can be run as follows:    
`python train.py train`  
`python train.py predict 1,2,3,4.......`  


## Requirement
```
python==3.7.0
numpy==1.18.1
```


## Notes   

I used 4 dense layers in my model, and MAE was used as a loss function.
The following table summarizes the global

Parameters:

* Learning rate = 0.002  
* Batch size = 32  
* Optimizer = MSprop  
* Fliters = 32  
* epochs = 100

Dataset |  Training MAE |  Training MSE  |  Validation MAE |  Validation MSE  |
:---: | :---:  | :---: | :---: | :---:
Wine quality | 0.2888 | 0.1553  | 0.4948 | 0.4444
