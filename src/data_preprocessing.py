import pandas as pd
import numpy as np
from sklearn.preprocessing import StandardScaler, OneHotEncoder
from sklearn.impute import SimpleImputer
from sklearn.compose import ColumnTransformer
from sklearn.pipeline import Pipeline

def load_data(file_path):
    return pd.read_csv(file_path)

def preprocess_data(df):
    # Select relevant features
    features = ['tier', 'flagship', 'public', 'par_income_bin', 'rel_apply', 'rel_attend', 'rel_att_cond_app', 'attend']
    X = df[features]
    y = (X['attend'] > X['attend'].median()).astype(int)  # Binary classification target
    X = X.drop('attend', axis=1)

    # Define numeric and categorical columns
    numeric_features = ['rel_apply', 'rel_attend', 'rel_att_cond_app']
    categorical_features = ['tier', 'flagship', 'public', 'par_income_bin']

    # Create preprocessing pipelines
    numeric_transformer = Pipeline(steps=[
        ('imputer', SimpleImputer(strategy='median')),
        ('scaler', StandardScaler())
    ])

    categorical_transformer = Pipeline(steps=[
        ('imputer', SimpleImputer(strategy='constant', fill_value='missing')),
        ('onehot', OneHotEncoder(handle_unknown='ignore'))
    ])

    # Combine preprocessing steps
    preprocessor = ColumnTransformer(
        transformers=[
            ('num', numeric_transformer, numeric_features),
            ('cat', categorical_transformer, categorical_features)
        ])

    # Fit and transform the data
    X_processed = preprocessor.fit_transform(X)
    feature_names = (numeric_features +
                     preprocessor.named_transformers_['cat'].named_steps['onehot'].get_feature_names(categorical_features).tolist())

    return X_processed, y, feature_names, preprocessor

def split_data(X, y, test_size=0.2, random_state=42):
    from sklearn.model_selection import train_test_split
    return train_test_split(X, y, test_size=test_size, random_state=random_state, stratify=y)