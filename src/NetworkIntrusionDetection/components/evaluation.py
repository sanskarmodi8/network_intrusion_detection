from NetworkIntrusionDetection.entity.config_entity import EvaluationConfig
import joblib
import pandas as pd
from sklearn.metrics import accuracy_score, precision_score, recall_score, f1_score, silhouette_score
import mlflow
import mlflow.sklearn
from urllib.parse import urlparse
from NetworkIntrusionDetection.utils.common import save_json
from pathlib import Path
from NetworkIntrusionDetection import logger

class Evaluation:
    def __init__(self, config: EvaluationConfig):
        self.config = config
        self.df = pd.read_csv(self.config.test_data_path)
        self.X = self.df.drop(columns=["class"])
        self.y = self.df["class"]
        self.trained_model_paths = [
            ("isolation_forest_model", self.config.trained_isolation_forest_model_path),
            ("log_reg_model", self.config.trained_log_reg_model_path),
            ("decision_trees_model", self.config.trained_decision_trees_model_path),
            ("random_forest_model", self.config.trained_random_forest_model_path),
            ("xgboost_model", self.config.trained_xgboost_model_path),
            ("svm_model", self.config.trained_svm_model_path),
            ("naive_bayes_model", self.config.trained_naive_bayes_model_path),
            ("mlp_model", self.config.trained_mlp_model_path)
        ]
        
        # Initialize best_model as a tuple
        self.best_model = ("", None)
    
    def evaluate_and_log_into_mlflow(self, average='weighted'):
        mlflow.set_registry_uri(self.config.mlflow_uri)
        tracking_url_type_store = urlparse(mlflow.get_tracking_uri()).scheme
        
        self.best_score = 0
        
        with mlflow.start_run():
            mlflow.log_params(self.config.params)
            for model_name, model_path in self.trained_model_paths:
                
                model = joblib.load(model_path)
                y_pred = model.predict(self.X)
                accuracy = accuracy_score(self.y, y_pred)
                precision = precision_score(self.y, y_pred, average=average)
                recall = recall_score(self.y, y_pred, average=average)
                f1 = f1_score(self.y, y_pred, average=average)
                
                # Log evaluation metrics to MLflow
                mlflow.log_metric(f'{model_name}_accuracy', accuracy)
                mlflow.log_metric(f'{model_name}_precision', precision)
                mlflow.log_metric(f'{model_name}_recall', recall)
                mlflow.log_metric(f'{model_name}_f1', f1)
                
                score = {"metrics":{"accuracy":accuracy, "precision":precision, "recall": recall, "f1score":f1} }
                save_json(Path(f"{self.config.root_dir}/{model_name}_scores.json"), score)
                
                if f1 > self.best_score:
                    self.best_score = f1
                    self.best_model = (model_name, model)
                
                
                
                logger.info(f"\n\n{model_name} evaluated\n\n")
                
                if tracking_url_type_store != "file":
                    # Register the model
                    mlflow.sklearn.log_model(model, "model", registered_model_name=model_name)
                else:
                    mlflow.sklearn.log_model(model, "model")
    
    def evaluate_without_logging_in_mlflow(self, average='weighted'):
        self.best_score = 0
        
        for model_name, model_path in self.trained_model_paths:
            
            model = joblib.load(model_path)
            y_pred = model.predict(self.X)
            accuracy = accuracy_score(self.y, y_pred)
            precision = precision_score(self.y, y_pred, average=average)
            recall = recall_score(self.y, y_pred, average=average)
            f1 = f1_score(self.y, y_pred, average=average)
            
            score = {"metrics":{"accuracy":accuracy, "precision":precision, "recall": recall, "f1score":f1} }
            save_json(Path(f"{self.config.root_dir}/{model_name}_scores.json"), score)
            
            if f1 > self.best_score:
                self.best_score = f1
                self.best_model = (model_name, model)
                
            
                    
            logger.info(f"\n\n{model_name} evaluated\n\n")
    
    def save_best_model(self):
        model_name, model = self.best_model
        BEST_MODEL_PATH = Path(f"{self.config.best_model_dir}/best_model.h5")
        joblib.dump(model, BEST_MODEL_PATH)
        with open(self.config.best_model_name, "w") as f:
            f.write(model_name)
