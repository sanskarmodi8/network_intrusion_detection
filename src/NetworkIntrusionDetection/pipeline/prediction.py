import joblib
from NetworkIntrusionDetection.config.configuration import ConfigurationManager
import pandas as pd

class Prediction():
    def __init__(self):
        self.config = ConfigurationManager().getEvaluationConfig()
    
    def predict(self, data):
        model = joblib.load(self.config.best_model_path)
        data = pd.DataFrame(data,index=[0])
        print(data)
        result = model.predict(data)
        print(result)
        if result == 0:
            return "Normal"
        else:
            return "Intrusion"