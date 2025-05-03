import streamlit as st # type: ignore
import matplotlib.pyplot as plt # type: ignore
import pandas as pd # type: ignore
from sklearn.tree import DecisionTreeClassifier # type: ignore
from sklearn.metrics import precision_score, classification_report # type: ignore
from sklearn.model_selection import train_test_split # type: ignore


# Load dataset and train model
data = pd.read_csv("sleep_data.csv")
X = data[["Age", "Screen Time (hrs)", "Caffeine (mg)", "Exercise (mins)", "Bedtime"]]
y = data["Sleep Quality"]
model = DecisionTreeClassifier()
model.fit(X, y)

# Streamlit UI
st.title("😴 Sleep Quality Predictor")

# User input widgets
age = st.slider("Age", 18, 60)
screen_time = st.slider("Daily Screen Time (hours)", 0, 12)
caffeine = st.slider("Caffeine Intake (mg)", 0, 500)
exercise = st.slider("Exercise (minutes)", 0, 120)

# New bedtime input with AM/PM dropdown
col1, col2 = st.columns(2)
with col1:
    hour = st.slider("Hour", 1.0, 12.0, step=0.5, value=10.0)  # 1-12 hours
with col2:
    period = st.selectbox("AM/PM", ["PM", "AM"])  # Dropdown

# Convert to 24-hour decimal (model-compatible)
bedtime = hour + 12.0 if (period == "PM" and hour < 12.0) else hour  # 10 PM → 22.0
if period == "AM" and hour == 12.0:  # Handle 12 AM
    bedtime = 0.0

# Predict button
if st.button("Predict My Sleep Quality"):
    user_input = [[age, screen_time, caffeine, exercise, bedtime]]
    prediction = model.predict(user_input)[0]
    
    # Prediction results
    if prediction == 1:
        st.success("Good sleep quality! 🎉")
    else:
        st.error("Poor sleep quality! 💤")
        if screen_time > 5:
            st.warning("Try reducing screen time by 1 hour before bed!")
        elif caffeine > 200:
            st.warning("Your caffeine intake is high! Limit to <200mg daily.")
    
    # Visualization
    st.subheader("Sleep Quality Trends")
    fig, ax = plt.subplots()
    data.groupby("Bedtime")["Sleep Quality"].mean().plot(
        ax=ax, 
        title="How Bedtime Affects Sleep Quality",
        xlabel="Bedtime (24h format)", 
        ylabel="% Good Sleep"
    )
    # Highlight user's bedtime
    ax.axvline(x=bedtime, color='red', linestyle='--', label='Your bedtime')
    ax.legend()
    st.pyplot(fig)