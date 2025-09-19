
# Crypto Coin Volatility

## Problem statement

In this data science project, We will build a machine learning system which will be able predict the crypto coin volatility using machine learning algorithms. This project will be very usefull for crypto trader/investor to take decision accordingly. Based on Liquidity ratio(volume/market cap) we can make classification and then train classification models.

## Solution Proposed

How to make  classification for the coins ?. The best approch is to use liquidity ratio which represent how coin is being traded, basic cut off are if ratio is below 0.2 it is considered as low volatilie coin which should be avoided due to liquidity conserns if ranges between 0.2 - 0.12 can be considered as a stable coin and if above that it is highly volatilie coin which should be avoided due to price instablity conserns.

Dataset used
 <html>
<a href="https://drive.google.com/drive/folders/1qvXRekLJkdLwoI5dxb86OOx5KooklLGC"> Dataset Link</a>
</html>



## Tech Stack Used

1. Python
2. FastAPI
3. Machine learning algorithms

## Infrastructure required

* Local Host 

## How to run

Step 1. Cloning the repository.

```

git clone https://github.com/siddharthnahata/project_crypto.git

```

Step 2. Create a python environment.

```

python3 -m venv venv

```

```
myenv\Scripts\activate.bat

```

Step 3. Install the requirements

```

pip install -r requirements.txt

```

Step 4. Run the application server

```

python run.py

```

Step 5. Prediction application

```

http://localhost:5000/predict

```

## Models Used

* [XGBoost Classifier](https://xgboost.readthedocs.io/en/stable/)

**Components** : Contains all components of Machine Learning Project

- Data Ingestion
- Data Validation
- Data Transformation
- Model Trainer
- Model Evaluation
- Model Pusher

## Project Summary

1. We feteched the data from the above dataset link.
2. In notebook folder EDA.ipynb file we processed data like creating some new features and droping unnecessary features
3. Then we conducted Univarite, Bivariate and Multivariate analysis the noted observations.
4. After that we made src file which contain 2 scripts of which one is preprocessing.py which have function which handles the data and scaling of it.
5. Other script which is train.py which train the model with help of first script and store .pkl file of the model in modols folder.
6. Created app.py script which runs the model on local host.
7. Created run.py which is also way to run app.py with python command.

## Conclusion

- This Project can be used in real-life by Users.
