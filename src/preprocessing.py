import pandas as pd
import numpy as np
from sklearn.preprocessing import LabelEncoder, StandardScaler
from sklearn.compose import ColumnTransformer
from sklearn.pipeline import Pipeline
from catboost import CatBoostClassifier

import warnings
warnings.filterwarnings("ignore")


def preprocess(df: pd.DataFrame) -> pd.DataFrame:
    """
        This funtion pre process the data for the model building does not include scaling.
        It just having droping duplicated and nan values, makeing new features and droping not usefull columns.
    """

    print("Dropping null values.")
    df.dropna(inplace=True)

    print("Checking for duplicated values")
    duplicated_value_sum = df.duplicated().sum()
    print("Founded: ", duplicated_value_sum, "Values")
    if duplicated_value_sum:
        print(f"Dropping \n{df[df.duplicated()]}")
        df.drop_duplicates(inplace=True)
    else:
        print("Continuing since no duplicated value found...")

    # converting both columns to int
    print("Converting some values to int")
    col_to_convert = ['mkt_cap', '24h_volume']
    for col in col_to_convert:
        df[col] = df[col].astype(int)
        print(f"Converted {col} to int")

    # Droping the column as not required for the model
    print("Dropping unnecessary columns ")
    col_to_drop = ['coin', 'symbol', 'price', 'date']
    for col in col_to_drop:
        df.drop(col, axis=1, inplace=True)
        print(f"Dropped: {col}")

    print("Creating new feature called Liquidity ratio which represent the voltality of a particular")
    df['Liquidity_Ratio'] = df['24h_volume']/df['mkt_cap'] # this is basic formula used to calculate voltality score further we will make cut off to rate the volatility


    df.reset_index(drop=True, inplace=True)

    print("Making a target columns as we can see from liquid ratio")
    # basically if ratio is less than 0.02 it is low volatile and if ranges between 0.02 to 0.12 it is considered as perfect and if greated than that can be very dangerous in terms of price fluctuation
    df['volatility'] = pd.cut(
        df['Liquidity_Ratio'],
        bins=[-float('inf'), 0.02, 0.12, float('inf')],
        labels=['Low', 'Stable', 'High']
    )
    df.drop('Liquidity_Ratio', axis=1, inplace=True) # droping it due to it will cause biasness in the model as directly related to the coin volaitlty
    return df

def build_pipeline():
    """
        Build ML pipeline with preprocessing + model
    """
    scale_cols = ["24h_volume", "mkt_cap"]
    passthrough_cols = ["1h", "24h", "7d"]

    preprocessor = ColumnTransformer(
        transformers=[
            ("num", StandardScaler(), scale_cols),
            ("passthrough", "passthrough", passthrough_cols)
        ]
    )
    pipeline = Pipeline([
        ("preprocessor", preprocessor),
        ("model", CatBoostClassifier(depth=6, iterations=200, l2_leaf_reg=5, learning_rate=0.1))
    ])

    return pipeline

def encode_target(y: pd.Series):
    """
    Encode target labels into integers
    """
    le = LabelEncoder()
    y_encoded = le.fit_transform(y)
    return y_encoded, le