from fastapi import FastAPI
from fastapi.responses import JSONResponse
from fastapi.middleware.cors import CORSMiddleware
from NetworkIntrusionDetection.pipeline.prediction import Prediction
import os
from sklearn.preprocessing import StandardScaler, LabelEncoder
from pydantic import BaseModel
import uvicorn

app = FastAPI()

# Allow CORS
app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)

class Input(BaseModel):
    is_host_login: int
    num_outbound_cmds: int
    same_srv_rate: float
    dst_host_srv_count: int
    dst_host_same_srv_rate: float
    logged_in: int
    dst_host_srv_serror_rate: float
    dst_host_serror_rate: float
    serror_rate: float
    flag: str
    srv_serror_rate: float
    count: int
    dst_host_count: int
    service: str
    dst_host_rerror_rate: float
    
    class Config:
        json_schema_extra = {
            "example": {
                "is_host_login": 0,
                "num_outbound_cmds": 0,
                "same_srv_rate": 0.5,
                "dst_host_srv_count": 10,
                "dst_host_same_srv_rate": 0.6,
                "logged_in": 1,
                "dst_host_srv_serror_rate": 0.1,
                "dst_host_serror_rate": 0.2,
                "serror_rate": 0.3,
                "flag": "SF",
                "srv_serror_rate": 0.4,
                "count": 100,
                "dst_host_count": 255,
                "service": "http",
                "dst_host_rerror_rate": 0.1
            }
        }

class ClientApp:
    def __init__(self):
        self.classifier = Prediction()


@app.get("/")
async def home():
    return {"message": "Welcome to the Netwotk Intrusion detection API --by Sanskar Modi", "/train" : "go to this route to start the training pipeline", "/docs" : "go to this route to be able to send post request on route /predict for classification"}

@app.get("/train")
async def trainRoute():
    # os.system("python main.py")
    os.system("dvc repro")
    return "Training done successfully!"

@app.post("/predict")
async def predict_route(input: Input):
    try:
        # Create an instance of the Prediction class
        clApp = ClientApp()
        
        # Preprocess the input data
        data_dict = input.dict()
        
        # Numerical features for standard scaling
        numerical_features = ['same_srv_rate', 'dst_host_srv_count', 'dst_host_same_srv_rate',
                            'dst_host_srv_serror_rate', 'dst_host_serror_rate', 'serror_rate',
                            'srv_serror_rate', 'count', 'dst_host_count', 'dst_host_rerror_rate']
        
        # Categorical features for label encoding
        categorical_features = ['flag', 'service']
        
        # Standard scaling numerical features
        scaler = StandardScaler()
        for feature in numerical_features:
            data_dict[feature] = scaler.fit_transform([[data_dict[feature]]])[0][0]
        
        # Label encoding categorical features
        label_encoders = {}
        for feature in categorical_features:
            if feature in data_dict:
                label_encoders[feature] = LabelEncoder()
                data_dict[feature] = label_encoders[feature].fit_transform([data_dict[feature]])[0]
        
        # Make the prediction
        result = clApp.classifier.predict(data_dict)
        
        # Return the result
        return JSONResponse({"result": result})
    except Exception as e:
        print(e)
        return JSONResponse({"error": str(e)})

if __name__ == "__main__":
    
    uvicorn.run(app, host="0.0.0.0", port=8080)