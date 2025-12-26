from fastapi import FastAPI
from pydantic import BaseModel
from datetime import date

from optimizer import (
    load_and_validate_data,
    create_features,
    train_model,
    recommend_price
)

# ✅ CREATE APP FIRST
app = FastAPI(title="Fuel Price Optimization API")

# ----------------------------------
# LOAD DATA & TRAIN MODEL AT STARTUP
# ----------------------------------
DATA_PATH = "data/oil_retail_history.csv"

df_raw = load_and_validate_data(DATA_PATH)
df_feat = create_features(df_raw)
model, FEATURES = train_model(df_feat)

# ----------------------------------
# REQUEST SCHEMA
# ----------------------------------
class TodayInput(BaseModel):
    date: date
    price: float
    cost: float
    comp1_price: float
    comp2_price: float
    comp3_price: float

# ----------------------------------
# API ENDPOINT
# ----------------------------------
@app.post("/recommend-price")
def get_price(data: TodayInput):
    payload = data.dict()
    payload["date"] = payload["date"].isoformat()

    return recommend_price(
        today_data=payload,
        history_df=df_feat,
        model=model,
        FEATURES=FEATURES
    )
