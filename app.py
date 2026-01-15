# app.py

# --- Core Libraries for the Application ---
import streamlit as st
import pandas as pd
import joblib

# --- Caching and Loading the Model and Scaler ---
@st.cache_resource
def load_assets():
    """
    Loads the saved machine learning model and scaler from disk.
    This function is cached to ensure it's only run once per session.
    """
    scaler = joblib.load('scaler.joblib')
    model = joblib.load('final_model.joblib')
    return scaler, model

# --- Load the assets into the main part of the script ---
scaler, model = load_assets()

# --- NEW CODE STARTS HERE ---

# --- User Interface: Title and Introduction ---

# Use st.title() to add a main title to the web application.
# This is the first thing a user will see and it sets the context for the app.
# We're using an emoji for a bit of visual flair!
st.title('🏡 California Real Estate Price Predictor')


# --- NEW CODE STARTS HERE ---

# --- Sidebar for User Inputs ---

# st.sidebar.header() places a header in the sidebar panel on the left.
# This is the ideal place to organize all the user input widgets.
st.sidebar.header('User Input Features')


# --- NEW CODE STARTS HERE ---
def user_input_features():
    # For each feature, we create a widget. The value from the widget is stored in a variable.
    # We use st.sidebar to place these widgets in the sidebar.

    # st.sidebar.slider is great for features with a clear, constrained range.
    # Arguments: label, min_value, max_value, default_value
    house_age = st.sidebar.slider('House Age (years)', 1, 52, 25)

    # st.sidebar.number_input is better for features that can have more open-ended or precise decimal values.
    # Arguments: label, min_value, max_value, default_value, step
    med_inc = st.sidebar.number_input('Median Income (in tens of thousands of $)', 1.0, 15.0, 3.5, 0.1)

    # We create widgets for all 8 base features the model needs.
    # The default values are set to be reasonable median values for California.
    avg_rooms = st.sidebar.number_input('Average Number of Rooms', 2.0, 10.0, 5.0, 0.5)
    avg_bedrms = st.sidebar.number_input('Average Number of Bedrooms', 1.0, 5.0, 1.0, 0.5)
    population = st.sidebar.number_input('Block Population', 500, 5000, 1500, 100)
    avg_occup = st.sidebar.number_input('Average House Occupancy', 1.0, 10.0, 2.5, 0.25)
    latitude = st.sidebar.number_input('Latitude', 32.0, 42.0, 35.6, 0.1)
    longitude = st.sidebar.number_input('Longitude', -124.0, -114.0, -119.5, 0.1)

    # Store all the inputs in a dictionary. The keys must match the feature names
    # our model was trained on.
    data = {
        'HouseAge': house_age,
        'MedInc': med_inc,
        'AveRooms': avg_rooms,
        'AveBedrms': avg_bedrms,
        'Population': population,
        'AveOccup': avg_occup,
        'Latitude': latitude,
        'Longitude': longitude
    }

    return data

    # Call the function to get the user's input.
    # The returned dictionary is stored in the 'user_inputs' variable.


user_inputs = user_input_features()

# --- NEW CODE STARTS HERE ---

# --- Prepare User Inputs for the Model ---

# Convert the user's input dictionary into a pandas DataFrame.
# The first argument is the dictionary of data.
# The `index=[0]` argument is crucial; it tells pandas to create a DataFrame with a single row.
input_df = pd.DataFrame(user_inputs, index=[0])

# --- Feature Engineering: A Critical Step! ---
# Remember the Feature Engineering step in your notebook? You created new features
# like 'rooms_per_person' and 'bedrooms_per_room'.
# Your model was trained with these features, so you MUST create them for the new
# input data as well, using the exact same calculations.

# Create the 'rooms_per_person' feature.
# We add a small check to avoid division by zero, a good robust practice.
if input_df['Population'][0] > 0:
    input_df['rooms_per_person'] = input_df['AveRooms'][0] / input_df['Population'][0]
else:
    # If population is zero, this ratio is undefined. We can set it to 0 or another sensible default.
    input_df['rooms_per_person'] = 0

# Create the 'bedrooms_per_room' feature.
if input_df['AveRooms'][0] > 0:
    input_df['bedrooms_per_room'] = input_df['AveBedrms'][0] / input_df['AveRooms'][0]
else:
    input_df['bedrooms_per_room'] = 0

# It's important to ensure the final DataFrame has its columns in the exact same order
# as the one used for training the model. Let's define that order.
# You can get this from your original notebook (e.g., from X.columns).
final_feature_order = ['MedInc', 'HouseAge', 'AveRooms', 'AveBedrms', 'Population',
                       'AveOccup', 'Latitude', 'Longitude', 'rooms_per_person', 'bedrooms_per_room']

# Reorder the columns of our input DataFrame to match the training data.
input_df = input_df[final_feature_order]

# --- Display the Final Prepared Data ---
# This is great for debugging and for showing the user the final data
# that is being fed into the model.
st.subheader('Final Input Features for Prediction')
st.write(input_df)

# --- NEW CODE STARTS HERE ---

# --- Prediction Logic ---

# We create a button in the main area of the app.
# The `st.button()` function returns True when the button is clicked.
# We use this to create a conditional block of code.

# app.py

# --- (The rest of your code remains unchanged) ---


# --- Prediction Logic ---

# We create a button in the main area of the app.
# The code inside this 'if' block will only run when the user clicks the button.
if st.button('Predict Price'):
    # --- HIGHLIGHTED CHANGE: The code below is now inside the button's logic ---

    # Step 1: Scale the user's input data.
    # We use the .transform() method of our loaded scaler.
    # This applies the exact same scaling transformation that was applied to the training data.
    # The input must be a DataFrame or 2D array-like structure. Our input_df is perfect.
    scaled_input = scaler.transform(input_df)

    # (Optional, but highly recommended for debugging)
    # Display the scaled data to confirm the transformation worked.
    #st.subheader('Scaled Input Features')
    #st.write("These are the features after scaling, which will be fed to the model:")
    #st.write(scaled_input)

    # Placeholder for the next step
    #st.write('Next step: Make prediction using the model...')

    # Step 2: Make a prediction using the loaded model.
    # The .predict() method takes the scaled data and returns the model's prediction.
    # The result is a NumPy array, even if it's just a single value.
    prediction = model.predict(scaled_input)

    # The prediction is an array like [4.53], so we extract the single value from it.
    predicted_price = prediction[0]

    # (Placeholder for the final task)
    # We will display this value in the next step. For now, let's just write it
    # out to confirm it's working.
    st.write("---")
    #st.write(f"The raw prediction value from the model is: {predicted_price}")
    #st.write("This value will be formatted and displayed nicely in the final step.")

# This replaces the placeholder st.write() calls from the previous step.

    # Step 3: Display the final prediction to the user.

    # First, convert the model's output (in hundreds of thousands) to an actual dollar value.
    final_price = predicted_price * 100000
    # Use st.success() to display the result in a visually appealing green box.
    # We use an f-string to format the output nicely.
    #   - The '$' sign is added for currency.
    #   - The {final_price:,.0f} part is a format specifier:
    #     - `:` starts the format spec.
    #     - `,` adds a comma as a thousands separator.
    #     - `.0f` formats the number as a float with zero decimal places.
    st.success(f'The predicted median house price is: ${final_price:,.0f}')
