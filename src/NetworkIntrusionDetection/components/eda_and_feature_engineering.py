from NetworkIntrusionDetection import logger
from NetworkIntrusionDetection.entity.config_entity import FEConfig
from sklearn.feature_selection import VarianceThreshold, SelectKBest, f_classif
from sklearn.preprocessing import StandardScaler, LabelEncoder
from sklearn.model_selection import train_test_split
import pandas as pd
import numpy as np


class FeatureEngineering:
    def __init__(self, config: FEConfig):
        self.config = config

    def fe(self):
        
        logger.info("Feature Engineering started")

        df = pd.read_csv(self.config.data_file)
        
        # encode the target feature 
        df["class"] = df["class"].replace({"normal": 0, "anomaly": 1})
        
        # train and test split
        X_train, X_test, y_train, y_test = train_test_split(df.drop('class', axis=1), df['class'], test_size=0.2, random_state=42)
        
        # num and cat data
        num_cols = X_train.select_dtypes(exclude="object").columns
        cat_cols = X_train.select_dtypes(include="object").columns
        
        # convert X data to dataframe format
        X_train = pd.DataFrame(X_train, columns=X_train.columns)
        X_test = pd.DataFrame(X_test, columns=X_test.columns)
        
        # apply standard scaling to num data
        scaler = StandardScaler()
        X_train[num_cols] = scaler.fit_transform(X_train[num_cols])
        X_test[num_cols] = scaler.transform(X_test[num_cols])
        
        # Define all possible categories based on the unique values in both training and test sets for all categorical columns
        all_categories = set(X_train["service"].unique()).union(set(X_test["service"].unique()))
        all_categories = all_categories.union(set(X_train["protocol_type"].unique())).union(set(X_test["protocol_type"].unique()))
        all_categories = all_categories.union(set(X_train["flag"].unique())).union(set(X_test["flag"].unique()))

        # Initialize LabelEncoder
        le = LabelEncoder()

        # Fit LabelEncoder on all possible categories
        le.fit(list(all_categories))

        # Transform categorical columns in both training and test sets
        for col in cat_cols:
            X_train[col] = le.transform(X_train[col])
            X_test[col] = le.transform(X_test[col])

        temp = X_train
        
        # apply select k best
        selector = SelectKBest(f_classif, k=self.config.select_k_best)
        X_train = selector.fit_transform(X_train, y_train)
        X_test = selector.transform(X_test)
        
        # get the selected features names 
        feature_importances = selector.scores_
        top_k_indices = np.argsort(feature_importances)[::-1][:self.config.select_k_best]  # Descending order, top k

        # Select features based on the top k indices
        selected_features = temp.columns[top_k_indices]
        
        # save the final data
        train = pd.DataFrame(X_train, columns=selected_features)
        test = pd.DataFrame(X_test, columns=selected_features)
        y_train.reset_index(drop=True, inplace=True)
        y_test.reset_index(drop=True, inplace=True)
        train['class'] = y_train
        test['class'] = y_test
        train.to_csv(self.config.final_data_train, index=False)
        test.to_csv(self.config.final_data_test, index=False)
        logger.info(f"Feature Engineering completed and saved the final data at {self.config.final_data_train} and {self.config.final_data_test}")
        
        
            

