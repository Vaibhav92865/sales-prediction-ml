from fastapi import FastAPI
from fastapi.middleware.cors import CORSMiddleware
from pydantic import BaseModel
import joblib
import pandas as pd
import numpy as np


app = FastAPI()


# Allow frontend to connect with API
app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)


# Load trained model
model = joblib.load("model/sales_model.pkl")

# Load encoder
encoder = joblib.load("model/encoder.pkl")

# Load feature information
features = joblib.load("model/features.pkl")


# Input data schema
class SalesInput(BaseModel):
    Branch: str
    City: str
    Customer_type: str
    Gender: str
    Product_line: str
    Payment: str

    Unit_price: float
    Quantity: int
    Tax_5: float
    Total: float
    cogs: float
    gross_margin_percentage: float
    gross_income: float
    Rating: float

    Month: int
    Year: int
    DayOfWeek: int


@app.get("/")
def home():
    return {
        "message": "Sales Prediction API is running"
    }


@app.post("/predict")
def predict(data: SalesInput):

    # Convert input into dictionary
    input_data = data.model_dump()

    # Convert API field names to dataset column names
    input_data = {
        "Branch": input_data["Branch"],
        "City": input_data["City"],
        "Customer type": input_data["Customer_type"],
        "Gender": input_data["Gender"],
        "Product line": input_data["Product_line"],
        "Payment": input_data["Payment"],
        "Unit price": input_data["Unit_price"],
        "Quantity": input_data["Quantity"],
        "Tax 5%": input_data["Tax_5"],
        "Total": input_data["Total"],
        "cogs": input_data["cogs"],
        "gross margin percentage":
            input_data["gross_margin_percentage"],
        "gross income": input_data["gross_income"],
        "Rating": input_data["Rating"],
        "Month": input_data["Month"],
        "Year": input_data["Year"],
        "DayOfWeek": input_data["DayOfWeek"]
    }

    # Create DataFrame
    input_df = pd.DataFrame([input_data])

    # Get feature names
    categorical_cols = features["categorical_cols"]
    numeric_cols = features["numeric_cols"]

    # Encode categorical data
    encoded_data = encoder.transform(
        input_df[categorical_cols]
    )

    # Get numeric data
    numeric_data = input_df[numeric_cols].values

    # Combine numeric + encoded data
    final_data = np.hstack([
        numeric_data,
        encoded_data
    ])

    # Make prediction
    prediction = model.predict(final_data)[0]

    return {
        "predicted_sales": round(float(prediction), 2)
    }