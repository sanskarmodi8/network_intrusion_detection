from NetworkIntrusionDetection.entity.config_entity import PrepareModelConfig
from sklearn.cluster import DBSCAN
from sklearn.ensemble import IsolationForest
from sklearn.neighbors import LocalOutlierFactor
from sklearn.linear_model import LogisticRegression
from sklearn.tree import DecisionTreeClassifier
from sklearn.ensemble import RandomForestClassifier
from xgboost import XGBClassifier
from sklearn.svm import SVC
from sklearn.naive_bayes import GaussianNB
from sklearn.ensemble import BaggingClassifier
from sklearn.neural_network import MLPClassifier
import joblib

class PrepareModel:
    def __init__(self, config: PrepareModelConfig):
        self.config = config
        
    def get_models(self):
        
        self.isolation_forest_model = self.isolation_forest_model(self.config.params.n_estimators_isolation, self.config.params.bootstrap)
        self.log_reg_model = self.log_reg_model(self.config.params.penalty, self.config.params.solver, self.config.params.max_iter_logreg, self.config.params.l1_ratio)
        self.decision_trees_model = self.decision_trees_model()
        self.random_forest_model = self.random_forest_model(self.config.params.n_estimators_random_forest, self.config.params.max_depth_random_forest, self.config.params.criterion, self.config.params.bootstrap_random_forest)
        self.xgboost_model = self.xgboost_model(self.config.params.n_estimators_xgboost, self.config.params.max_depth_xgboost, self.config.params.learning_rate_xgboost)
        self.svm_model = self.svm_model()
        self.naive_bayes_model = self.naive_bayes_model()
        self.ann_model = self.ann_model(self.config.params.hidden_layer_sizes, self.config.params.hidden_layer_activation, self.config.params.optimizer, self.config.params.l2, self.config.params.batch_size, self.config.params.learning_rate, self.config.params.learning_rate_init, self.config.params.EPOCHS, self.config.params.tol, self.config.params.early_stopping, self.config.params.n_iter_no_change)
    
        model_paths = [self.config.isolation_forest_model_path, self.config.log_reg_model_path, self.config.decision_trees_model_path, self.config.random_forest_model_path, self.config.xgboost_model_path, self.config.svm_model_path, self.config.naive_bayes_model_path, self.config.mlp_model_path]
        models = [self.isolation_forest_model, self.log_reg_model, self.decision_trees_model, self.random_forest_model, self.xgboost_model, self.svm_model, self.naive_bayes_model, self.ann_model]
        self.save_models(models, model_paths)

    @staticmethod
    def isolation_forest_model(n_estimators_isolation:int, bootstrap:bool):
        return IsolationForest(n_estimators=n_estimators_isolation, bootstrap=bootstrap)
    @staticmethod
    def log_reg_model(penalty:str, solver: str, max_iter_logreg:int, l1_ratio:float):
        return LogisticRegression(penalty=penalty, solver=solver, max_iter=max_iter_logreg, l1_ratio=l1_ratio)
    @staticmethod
    def decision_trees_model():
        return DecisionTreeClassifier()
    @staticmethod
    def random_forest_model(n_estimators_random_forest:int, max_depth_random_forest:int, criterion:str, bootstrap_random_forest:bool):
        return RandomForestClassifier(n_estimators=n_estimators_random_forest, max_depth=max_depth_random_forest, criterion=criterion, bootstrap=bootstrap_random_forest)
    @staticmethod
    def xgboost_model(n_estimators_xgboost:int, max_depth_xgboost:int, learning_rate_xgboost:float):
        return XGBClassifier(n_estimators=n_estimators_xgboost, max_depth=max_depth_xgboost, learning_rate=learning_rate_xgboost)
    @staticmethod
    def svm_model():
        return SVC()
    @staticmethod
    def naive_bayes_model():
        return GaussianNB()
    @staticmethod
    def ann_model(hidden_layer_sizes:int, hidden_layer_activation:str, optimizer:str, l2:float, batch_size:int, learning_rate:str, learning_rate_init:float, epochs:int, tol:float, early_stopping:bool, n_iter_no_change:int):
        return MLPClassifier(hidden_layer_sizes=hidden_layer_sizes, activation=hidden_layer_activation, solver=optimizer, alpha=l2, batch_size=batch_size, learning_rate=learning_rate, learning_rate_init=learning_rate_init, max_iter=epochs, tol=tol, early_stopping=early_stopping, n_iter_no_change=n_iter_no_change)
    @staticmethod
    def save_models(models:list, model_paths:list):
        for model in models:
            joblib.dump(model, model_paths[models.index(model)])