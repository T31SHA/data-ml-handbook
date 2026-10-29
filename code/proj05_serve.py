# HIDE
import warnings; warnings.filterwarnings("ignore")
# ENDHIDE
import joblib, pandas as pd
from fastapi import FastAPI
from fastapi.testclient import TestClient
from pydantic import BaseModel, Field

model = joblib.load("data/house_model.joblib")
app = FastAPI(title="house-price")
class Block(BaseModel):
    MedInc: float = Field(gt=0); HouseAge: float; AveRooms: float; AveBedrms: float
    Population: float; AveOccup: float; Latitude: float; Longitude: float; region: str
@app.post("/predict")
def predict(b: Block):
    yhat = float(model.predict(pd.DataFrame([b.model_dump()]))[0])
    return {"median_value_usd": round(yhat * 100_000), "model_version": "lgbm-v1"}

client = TestClient(app)                                  # in-process test, no server needed
r = client.post("/predict", json=dict(MedInc=8.3, HouseAge=41, AveRooms=6.98, AveBedrms=1.02, Population=322,
                                     AveOccup=2.55, Latitude=37.88, Longitude=-122.23, region="NorCal"))
print(r.status_code, r.json())
bad = client.post("/predict", json=dict(MedInc=-1, HouseAge=41, AveRooms=6.98, AveBedrms=1.02, Population=322,
                                       AveOccup=2.55, Latitude=37.88, Longitude=-122.23, region="NorCal"))
print(bad.status_code, "validation error ->", bad.json()["detail"][0]["msg"])
