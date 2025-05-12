# ===== IMPORTS WITH ERROR HANDLING =====
try:
    import streamlit as st
    import matplotlib.pyplot as plt
    import pandas as pd
    from sklearn.tree import DecisionTreeClassifier
    from sklearn.model_selection import train_test_split
    import os
except ImportError as e:
    import sys
    print(f"Critical import error: {e}", file=sys.stderr)
    sys.exit(1)

# ===== FILE VERIFICATION =====
@st.cache_resource
def load_data():
    try:
        if not os.path.exists("sleep_data.csv"):
            st.error("❌ Dataset file (sleep_data.csv) not found in:")
            st.code(os.path.abspath("."))
            st.stop()
        
        data = pd.read_csv("sleep_data.csv")
        required_columns = ["Age", "Screen Time (hrs)", "Caffeine (mg)", 
                          "Exercise (mins)", "Bedtime", "Sleep Quality"]
        
        if not all(col in data.columns for col in required_columns):
            missing = [col for col in required_columns if col not in data.columns]
            st.error(f"Missing columns in dataset: {', '.join(missing)}")
            st.stop()
            
        return data
    except Exception as e:
        st.error(f"Data loading failed: {str(e)}")
        st.stop()

# ===== MODEL TRAINING =====
@st.cache_resource
def train_model():
    data = load_data()
    X = data[["Age", "Screen Time (hrs)", "Caffeine (mg)", "Exercise (mins)", "Bedtime"]]
    y = data["Sleep Quality"]
    
    X_train, X_test, y_train, y_test = train_test_split(
        X, y, test_size=0.2, random_state=42
    )
    
    model = DecisionTreeClassifier()
    model.fit(X_train, y_train)
    
    return model

# ===== STREAMLIT UI =====
def main():
    st.title("😴 Sleep Quality Predictor")
    
    # Load model
    model = train_model()
    
    # User inputs
    col1, col2 = st.columns(2)
    with col1:
        age = st.slider("Age", 18, 60, 25)
        screen_time = st.slider("Screen Time (hours)", 0.0, 12.0, 4.5, 0.5)
        caffeine = st.slider("Caffeine (mg)", 0, 500, 100)
    with col2:
        exercise = st.slider("Exercise (mins)", 0, 120, 30)
        hour = st.slider("Hour", 1.0, 12.0, 10.0, 0.5)
        period = st.selectbox("AM/PM", ["PM", "AM"], index=0)
    
    # Time conversion
    bedtime = hour + 12.0 if (period == "PM" and hour < 12.0) else hour
    bedtime = 0.0 if (period == "AM" and hour == 12.0) else bedtime
    
    # Prediction
    if st.button("Predict Sleep Quality", type="primary"):
        user_input = [[age, screen_time, caffeine, exercise, bedtime]]
        prediction = model.predict(user_input)[0]
        
        if prediction == 1:
            st.success("Good sleep quality! 🎉")
            st.balloons()
        else:
            st.error("Poor sleep quality! 💤")
            if screen_time > 5:
                st.warning("👉 Reduce screen time before bed")
            if caffeine > 200:
                st.warning("👉 Lower caffeine intake (<200mg)")
            if bedtime > 23.0:
                st.warning("👉 Try earlier bedtime")
        
        # Visualization
        st.subheader("Sleep Patterns Analysis")
        try:
            data = load_data()
            fig, ax = plt.subplots(figsize=(10, 4))
            data.groupby("Bedtime")["Sleep Quality"].mean().plot(
                ax=ax, color='teal', linewidth=2.5
            )
            ax.axvline(bedtime, color='red', linestyle='--', label='Your bedtime')
            ax.set_title("Sleep Quality by Bedtime", pad=20)
            ax.set_xlabel("Bedtime (24h format)", labelpad=10)
            ax.set_ylabel("Quality Score", labelpad=10)
            ax.legend()
            ax.grid(alpha=0.3)
            st.pyplot(fig)
        except Exception as e:
            st.error(f"Error generating visualization: {str(e)}")

if __name__ == "__main__":
    main()
