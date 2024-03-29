
from NetworkIntrusionDetection import logger
from NetworkIntrusionDetection.entity.config_entity import FEConfig
from sklearn.feature_selection import VarianceThreshold, SelectKBest, f_classif
from sklearn.preprocessing import StandardScaler, LabelEncoder
import pandas as pd
import numpy as np

class FeatureEngineering:
    def __init__(self, config: FEConfig):
        self.config = config

    def fe(self):
        try:
            df = pd.read_csv(self.config.data_file)
            logger.info(
                f"Transforming the data saved at {self.config.data_file}")

            # Preprocessing
            num_cols = df.select_dtypes(include=['int64', 'float64']).columns
            df[num_cols] = StandardScaler().fit_transform(df[num_cols])
            cat_cols = ['class', 'protocol_type', 'service', 'flag']
            for col in cat_cols:
                df[col] = LabelEncoder().fit_transform(df[col])
            df2 = df.drop(columns=['class'])
            # Feature selection (adjust parameters as needed)
            selector = VarianceThreshold(
                threshold=self.config.variance_threshold)
            transformed_data = selector.fit_transform(df2)
            selector = SelectKBest(f_classif, k=self.config.select_k_best)
            transformed_data = selector.fit_transform(
                transformed_data, df['class'])
            
            feature_importances = selector.scores_
            # Get indices of top 10 features by descending importance
            top_10_indices = np.argsort(feature_importances)[::-1][:10]  # Descending order, top 10

            # Select features based on the top 10 indices
            selected_features = df.columns[top_10_indices]

            # merge the transformed data with the class column
            df = pd.concat([pd.DataFrame(
                transformed_data, columns=selected_features), df['class']], axis=1)
            # save
            df.to_csv(self.config.final_data_file, index=False)
            logger.info(
                f"Transformed data saved at {self.config.final_data_file}")
        except Exception as e:
            logger.error(f"Error during feature engineering: {e}")
