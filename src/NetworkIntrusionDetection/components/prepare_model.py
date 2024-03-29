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

class PrepareModel:
    def __init__(self, config: PrepareModelConfig):
        self.config = config

    def db_scan_model(self):
        pass
    
    def isolation_forest_model(self):
        pass
    
    def lof_model(self):
        pass
    
    def log_reg_model(self):
        pass
    
    def decision_trees_model(self):
        pass
    
    def random_forest_model(self):
        pass
    
    def xgboost_model(self):
        pass
    
    def svm_model(self):
        pass
    
    def naive_bayes_model(self):
        pass
    
    def custom_bagging_model(self):
        pass
    
    def ann_model(self):
        pass