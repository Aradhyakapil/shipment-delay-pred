from fastapi import FastAPI, HTTPException
from pydantic import BaseModel
import pandas as pd
import pickle
from sklearn.preprocessing import LabelEncoder

app = FastAPI(title="Shipment Delay Prediction API")

# Load the trained model
with open('/content/drive/MyDrive/best_model.pkl', 'rb') as f:
    model = pickle.load(f)

# Load the label encoders to transform user inputs to the correct format
with open('/content/drive/MyDrive/label_encoders.pkl', 'rb') as f:
    label_encoders = pickle.load(f)

class ShipmentDetails(BaseModel):
    Origin: str
    Destination: str
    Vehicle_Type: str
    Distance_km: float
    Weather_Conditions: str
    Traffic_Conditions: str

@app.post("/predict")
async def predict_delay(shipment_details: ShipmentDetails):
    try:
        # Convert pydantic input to dictionary
        input_dict = shipment_details.model_dump()
        # Convert the input values based on the label encoders used during training
        input_df = pd.DataFrame([input_dict])
        for col in ['Origin', 'Destination', 'Vehicle_Type', 'Weather_Conditions', 'Traffic_Conditions']:
            input_df[col] = label_encoders[col].transform(input_df[col])

        # Prepare input for the model
        input_features = input_df[['Origin', 'Destination', 'Vehicle_Type', 'Distance_km', 'Weather_Conditions', 'Traffic_Conditions']]

        # Make the prediction
        prediction = model.predict(input_features)[0]

        # Reverse the label encoding for output
        if prediction == 1:
          prediction_output = 'Delayed'
        else:
          prediction_output = 'On Time'
        return {"prediction": prediction_output}
    except Exception as e:
        raise HTTPException(status_code=500, detail=str(e))


@app.get("/")
async def root():
    return {"message": "Welcome to Shipment Delay Prediction API!"}