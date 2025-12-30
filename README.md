# Fuel Price Optimization Project

## About the Project
This project is about deciding the best daily fuel price using data.  
The goal is to choose a price that can give **maximum profit**, not just more sales.

The system uses past fuel price data to understand how price, cost, and competitor prices affect fuel demand. Based on this learning, it recommends the most suitable price for the day.

---

## What this project does
- Reads historical fuel retail data
- Cleans and prepares the data
- Predicts expected fuel demand using machine learning
- Tries different price options and calculates expected profit
- Recommends the price that is most likely to give the highest profit

---

## Technologies used
- Python
- Pandas & NumPy
- XGBoost
- FastAPI

---

## Project files
- `app.py` – API to get daily price recommendation
- `optimizer.py` – data processing, ML model, and price logic
- `data/` – historical data and sample input
- `requirements.txt` – required libraries

---

## How to run
1. Create virtual environment  
   ```bash python -m venv venv
   venv\Scripts\activate


  # Install dependencies
pip install -r requirements.txt


# Run the application

uvicorn app:app --reload


# Open browser and go to

http://127.0.0.1:8000/docs
   
# Example input
{
  "date": "2024-07-01",
  "price": 100,
  "cost": 92,
  "comp1_price": 99.5,
  "comp2_price": 100.5,
  "comp3_price": 101
}

# Example output
{
  "recommended_price": 101.15,
  "expected_volume": 4698.2,
  "expected_profit": 9510.4
}

# Note

The output values are expected values, not exact future results.
This system is meant to support pricing decisions using data.





