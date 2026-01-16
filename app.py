# app.py

# -------------------- Imports --------------------
import streamlit as st
import pandas as pd

from sklearn.datasets import fetch_california_housing
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler
from sklearn.ensemble import RandomForestRegressor


# -------------------- Model Training (CACHED) --------------------
@st.cache_resource
def train_model():
    # Load California Housing dataset from sklearn
    housing = fetch_california_housing(as_frame=True)
    df = housing.frame

    # Feature Engineering (same as training notebook)
    df["rooms_per_person"] = df["AveRooms"] / df["Population"]
    df["bedrooms_per_room"] = df["AveBedrms"] / df["AveRooms"]

    # Features and target
    X = df.drop("MedHouseVal", axis=1)
    y = df["MedHouseVal"]

    # Train-test split
    X_train, _, y_train, _ = train_test_split(
        X, y, test_size=0.2, random_state=42
    )

    # Scaling
    scaler = StandardScaler()
    X_train_scaled = scaler.fit_transform(X_train)

    # Model
    model = RandomForestRegressor(
        n_estimators=100,
        random_state=42,
        n_jobs=-1
    )
    model.fit(X_train_scaled, y_train)

    return scaler, model


# Load trained assets
scaler, model = train_model()


# -------------------- UI --------------------
st.title("Hi! I am akshara. This is my first end to end ML project - so please don't judge 🙏🏻 ")
st.title("🏡 California Real Estate Price Predictor")
st.write("Enter property details to predict the **median house price**.")


# -------------------- Sidebar Inputs --------------------
st.sidebar.header("User Input Features")

def user_input_features():
    house_age = st.sidebar.slider("House Age (years)", 1, 52, 25)
    med_inc = st.sidebar.number_input(
        "Median Income (in tens of thousands of $)", 1.0, 15.0, 3.5, 0.1
    )
    avg_rooms = st.sidebar.number_input("Average Rooms", 2.0, 10.0, 5.0, 0.5)
    avg_bedrms = st.sidebar.number_input("Average Bedrooms", 1.0, 5.0, 1.0, 0.5)
    population = st.sidebar.number_input("Population", 500, 5000, 1500, 100)
    avg_occup = st.sidebar.number_input("Average Occupancy", 1.0, 10.0, 2.5, 0.25)
    latitude = st.sidebar.number_input("Latitude", 32.0, 42.0, 35.6, 0.1)
    longitude = st.sidebar.number_input("Longitude", -124.0, -114.0, -119.5, 0.1)

    data = {
        "MedInc": med_inc,
        "HouseAge": house_age,
        "AveRooms": avg_rooms,
        "AveBedrms": avg_bedrms,
        "Population": population,
        "AveOccup": avg_occup,
        "Latitude": latitude,
        "Longitude": longitude,
    }

    return data


user_inputs = user_input_features()


# -------------------- Prepare Input Data --------------------
input_df = pd.DataFrame(user_inputs, index=[0])

# Feature engineering (same as training)
input_df["rooms_per_person"] = (
    input_df["AveRooms"] / input_df["Population"]
    if input_df["Population"][0] > 0
    else 0
)

input_df["bedrooms_per_room"] = (
    input_df["AveBedrms"] / input_df["AveRooms"]
    if input_df["AveRooms"][0] > 0
    else 0
)

# Correct column order
final_feature_order = [
    "MedInc",
    "HouseAge",
    "AveRooms",
    "AveBedrms",
    "Population",
    "AveOccup",
    "Latitude",
    "Longitude",
    "rooms_per_person",
    "bedrooms_per_room",
]

input_df = input_df[final_feature_order]

st.subheader("Prepared Input Features")
st.write(input_df)


# -------------------- Prediction --------------------
if st.button("Predict Price"):
    scaled_input = scaler.transform(input_df)
    prediction = model.predict(scaled_input)

    final_price = prediction[0] * 100000

    st.success(
        f"🏠 **Predicted House Price:** ${final_price:,.0f}"
    )
