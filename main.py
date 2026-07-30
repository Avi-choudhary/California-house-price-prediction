import io
import joblib
import pandas as pd
from fastapi import FastAPI, HTTPException, UploadFile,File
from fastapi.responses import StreamingResponse
from pydantic import BaseModel, Field

app=FastAPI()

model=joblib.load("house_model.joblib")
feature=joblib.load("house_features.joblib")

#input_Schema
class HouseFeatures(BaseModel):
    MedInc : float = Field(gt=0, description="median income of the house")
    HouseAge : float = Field(gt=0, description="Avg age of the house in the neighbourhood")
    AvgRoom : float = Field(gt=0, description="Avg room in the neighbourhood")
    AvgBedroom : float = Field(gt=0, description="Avg bedroom in the neighbourhood")
    Population : float = Field(gt=0, description="Avg population in the neighbourhood")
    AvgOccup : float = Field(gt=0, description="Avg no of the occupation in the neighbouhood")
    Latitude : float = Field(ge=32, le=42,description="Latitude of the location")
    Longitude : float = Field(ge=-125, le=-114, description="Longititude of the location")

#home
@app.get("/")
def home():
    return{
        "message" : "california house prediction api",
        "status" : "active",
        "endpoint" : "send POST request to /predict"
    }

@app.get("/health")
def health():
    return{
        "status" : "running",
        "model" : "RandomforestRegressor",
        "features" : feature,
        "Avg_error"  : "$39,000"
    }

#predicton
@app.post("/predict")
def predict(house : HouseFeatures):
    try:
        input_data = pd.DataFrame([{
            "MedInc": house.MedInc,
            "HouseAge": house.HouseAge,
            "AveRooms": house.AvgRoom,       # Changed from AvgRoom
            "AveBedrms": house.AvgBedroom,   # Changed from AvgBedroom
            "Population": house.Population,  # Fixed typo from Populaiton
            "AveOccup": house.AvgOccup,      # Changed from AvgOccup
            "Latitude": house.Latitude,
            "Longitude": house.Longitude
        }])

        predicted = model.predict(input_data)[0]
        price_usd = predicted*1000

        return{
            "Predicted_price" : f"${price_usd:,.0f}",
            "Predicted_price_short" : f"${predicted:.2f} houndred thoussand",
            "fidence_range" : f"${price_usd-39000:,.0f} to {price_usd+39000:,.0f}"
        }

    except Exception as e:
        raise HTTPException(
            status_code=500,
            detail=f"prediction failed : {str(e)}"
        )

@app.post("/predict-file")
async def predict_file(file:UploadFile=File(...)):
    if not file.filename.endswith(".csv"):
        raise HTTPException(
            status_code=400,
            detail="please upload a csv file only"
        )
    contents = await file.read()
    df=pd.read_csv(io.BytesIO(contents))

    required_columns = [
        "MedInc","HouseAge","AveRooms","AveBedrms","Population","AveOccup","Latitude","Longitude"
    ]
    missing_colums = [
        col for col in required_columns
        if col not in df.columns
    ]

    if missing_colums:
        raise HTTPException(
            status_code=400,
            detail=f"these columns are {missing_colums} missing from your file"
        )

    if len(df) == 00:
        raise HTTPException(
            status_code=400,
            detail="the file uploaded has no rows"
        )
    try:
        predictions = model.predict(df[required_columns])

        df["predicted_columns_usd"] = predictions
        df["predicted_columns_usd"] = df["predicted_columns_usd"].apply(lambda x: f"${x:,.0f}")

        output = df.to_csv(index=False)

        return StreamingResponse(
            io.StringIO(output),
            media_type="text/csv",
            headers={
                "content-desposition" : "attachement; filename=prediction.csv"
            }
        )
    except Exception as e:
        raise HTTPException(
            status_code=500,
            detail=f"prediction failed {str(e)}"

        )
    