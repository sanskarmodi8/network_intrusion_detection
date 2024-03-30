from NetworkIntrusionDetection.entity.config_entity import (TrainingConfig, PrepareModelConfig)
import joblib
import pandas as pd
from NetworkIntrusionDetection import logger

class Training:
    def __init__(self, config: TrainingConfig, preparemodel_config: PrepareModelConfig):
        self.config = config
        self.preparemodel_config = preparemodel_config
        
        #load the data
        train_data = pd.read_csv(self.config.data_train)
        
        #split the data
        self.X_train, self.y_train = train_data.drop('class', axis=1), train_data['class']
    
    def train_and_save_db_scan_model(self):
        model = joblib.load(self.preparemodel_config.db_scan_model_path)
        logger.info("Training DBSCAN model")
        model.fit(self.X_train, self.y_train)
        joblib.dump(model, self.config.trained_db_scan_model_path)
        logger.info("DBSCAN model trained and saved")
    
    def train_and_save_isolation_forest_model(self):
        model = joblib.load(self.preparemodel_config.isolation_forest_model_path)
        logger.info("Training Isolation Forest model")
        model.fit(self.X_train, self.y_train)
        joblib.dump(model, self.config.trained_isolation_forest_model_path)
        logger.info("Isolation Forest model trained and saved")
    
    def train_and_save_lof_model(self):
        model = joblib.load(self.preparemodel_config.lof_model_path)
        logger.info("Training LOF model")
        model.fit(self.X_train, self.y_train)
        joblib.dump(model, self.config.trained_lof_model_path)
        logger.info("LOF model trained and saved")
    
    def train_and_save_log_reg_model(self):
        model = joblib.load(self.preparemodel_config.log_reg_model_path)
        logger.info("Training Logistic Regression model")
        model.fit(self.X_train, self.y_train)
        joblib.dump(model, self.config.trained_log_reg_model_path)
        logger.info("Logistic Regression model trained and saved")
    
    def train_and_save_decision_trees_model(self):
        model = joblib.load(self.preparemodel_config.decision_trees_model_path)
        logger.info("Training Decision Trees model")
        model.fit(self.X_train, self.y_train)
        joblib.dump(model, self.config.trained_decision_trees_model_path)
        logger.info("Decision Trees model trained and saved")
    
    def train_and_save_random_forest_model(self):
        model = joblib.load(self.preparemodel_config.random_forest_model_path)
        logger.info("Training Random Forest model")
        model.fit(self.X_train, self.y_train)
        joblib.dump(model, self.config.trained_random_forest_model_path)
        logger.info("Random Forest model trained and saved")
    
    def train_and_save_xgboost_model(self):
        model = joblib.load(self.preparemodel_config.xgboost_model_path)
        logger.info("Training XGBoost model")
        model.fit(self.X_train, self.y_train)
        joblib.dump(model, self.config.trained_xgboost_model_path)
        logger.info("XGBoost model trained and saved")
    
    def train_and_save_svm_model(self):
        model = joblib.load(self.preparemodel_config.svm_model_path)
        logger.info("Training SVM model")
        model.fit(self.X_train, self.y_train)
        joblib.dump(model, self.config.trained_svm_model_path)
        logger.info("SVM model trained and saved")
    
    def train_and_save_naive_bayes_model(self):
        model = joblib.load(self.preparemodel_config.naive_bayes_model_path)
        logger.info("Training Naive Bayes model")
        model.fit(self.X_train, self.y_train)
        joblib.dump(model, self.config.trained_naive_bayes_model_path)
        logger.info("Naive Bayes model trained and saved")
    
    def train_and_save_mlp_model(self):
        model = joblib.load(self.preparemodel_config.mlp_model_path)
        logger.info("Training MLP model")
        model.fit(self.X_train, self.y_train)
        joblib.dump(model, self.config.trained_mlp_model_path)
        logger.info("MLP model trained and saved")
    