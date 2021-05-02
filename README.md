Shonit Gangoly 
[@ShoGan20](https://github.com/ShoGan20)   
 

> Video Demo: [YouTube]()  
> Requirements (I have been using these versions but you can use any):   
    `conda 4.10.1`<br> `jupyter core: 4.6.3` <br> `jupyter-notebook : 6.1.4`<br><br>
> Run:  
    
    1. Fork the repository and type jupyter notebook in anaconda console
    2. Select `Diabetes Prediction.ipynb` file to run in Jupyter Lab
    
> Libraries to import have been included in the ipynb file. They include:
> <br> `- Pandas`<br> `- Numpy`<br> `- sklearn` <br> `- imblearn`<br> `- seaborn` <br> `- matplotlib` <br> `- matplotlib.pyplot` <br>


## Introduction

The aim of the project is to use classification models on Prima Indians Diabetes Data set. It features two classification models k-Nearest Neighbors and Decision trees to predict diabetes in patients using 7 features as inputs.

## Methodology

   1. Using pd dataframes to read the diabetes.csv file.
   2. Performing Exploratory Data Analysis on the data set using `numpy` `matplotlib` and `seaborn`. 
   3. Use `seaborn` boxplots, crosstabs to find class correlation
   4. Split the dataset into training and testing set
   3. There is severe class imbalance in the data set. Using `sklearn.imblearn` to standardize and balance out the classes.
   4. Grid Search 5-fold cross validation to  estimate and tune the hyper parameters for the classification models.
   5. Use Principal Component Analysis and SMOTE to over sample the data set since it is small.
   6. Build k-Nearest Neighbors and Decision Tree models to classify the features
    . `roc_curve` to visualize the learning curve of the two models with different hyper-parameters and data set tuning.
    


## Functions

1. **Train-Test-Split**: Function to split the data set into 80% for training and 20% for testing.
2. **StandardScaler**: To bring all the features to the same scale in order to measure vector distances accurately for kNN
3. **Principal Component Analysis**: An algorithm to reduce the high level dimesnionality of the data set. This aids in reducing the amount of final calculations
4. **SMOTE**: Synthetic Minority Over-sampling TEchnique which artificially adds samples to the data set since the one I have used is small resulting in overfitting
5. **SimpleImputer**: To replace several Null values in the data set with the median values of the column.
6. **Grid Search Cross Validation**: Helps in estimating and tuning the hyper parameters of the classification models
7. **KNeighborsClassifier**: Estimates the wights of decision class by factoring in K Neighbors that have weights with least distances from it.
8. **DecisionTrees**: Creates decision trees to estimate the prediction path of output goal from root of the node to the child. 


## Data Set Used

- Prima Indians Diabetes Data Set
- Can be found here: [https://www.kaggle.com/uciml/pima-indians-diabetes-database](https://www.kaggle.com/uciml/pima-indians-diabetes-database)

## User Interface for Application

Prototype user interface created using [Proto.io](https://proto.io/)