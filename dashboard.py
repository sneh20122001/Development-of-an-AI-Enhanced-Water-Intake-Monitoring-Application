import streamlit as st
import pandas as pd
import numpy as np
from datetime import datetime, timedelta
from src.agent import WaterIntakeAgent
from src.database import log_intake, get_intake_history
from sklearn.linear_model import LinearRegression
import random

# Track whether user has started tracking
if "tracker_started" not in st.session_state:
    st.session_state.tracker_started = False


df = pd.DataFrame()
# ------------------------
# 💡 Welcome Section
# ------------------------
if not st.session_state.tracker_started:
    st.title("💧 Welcome to AI Water Tracker")
    st.markdown("""
        Track your daily hydration with the help of an AI assistant.  
        Log your intake, get smart feedback, and stay healthy effortlessly! 🧠💦
    """)

    if st.button("🚀 Start Tracking"):
        st.session_state.tracker_started = True
        st.experimental_rerun()

# ------------------------
# 🚰 Tracker Dashboard
# ------------------------
else:
    st.title("💧 AI Water Tracker Agent")

    # Sidebar: Intake Input & Profile Setup
    st.sidebar.header("Log Your Water Intake & Set Profile")
    user_id = st.sidebar.text_input("User ID", value="user_123")
    intake_ml = st.sidebar.number_input("Water Intake (ml)", min_value=0, step=100)
    
    # Set User Profile
    st.sidebar.subheader("Set Your Hydration Profile")
    weight = st.sidebar.number_input("Weight (kg)", min_value=30, step=1, value=70)
    age = st.sidebar.number_input("Age", min_value=18, step=1, value=25)
    activity_level = st.sidebar.selectbox("Activity Level", ["Low", "Medium", "High"])
    
    # Dynamic Hydration Goal Calculation
    hydration_goal = weight * 30  # base hydration in ml (30ml per kg)
    
    if activity_level == "Medium":
        hydration_goal *= 1.2
    elif activity_level == "High":
        hydration_goal *= 1.5

    st.sidebar.markdown(f"💧 **Your Hydration Goal**: {hydration_goal} ml/day")

    if st.sidebar.button("Submit"):
        if user_id and intake_ml:
            log_intake(user_id, intake_ml)
            st.success(f"✅ Logged {intake_ml}ml for {user_id}")
            
            # AI Water Intake Feedback
            agent = WaterIntakeAgent()
            feedback = agent.analyze_intake(intake_ml)
            st.info(f"🤖 AI Feedback: {feedback}")

    # Divider
    st.markdown("---")

    # 📊 History Section
    st.header("📈 Water Intake History")

    if user_id:
        history = get_intake_history(user_id)
        if history:
            dates = [datetime.strptime(row[1], "%Y-%m-%d") for row in history]
            values = [row[0] for row in history]

            df = pd.DataFrame({
                "Date": dates,
                "Water Intake (ml)": values
            })

            st.dataframe(df)
            st.line_chart(df, x="Date", y="Water Intake (ml)")

            # AI-Powered Prediction: Hydration Needs for Next 7 Days
            if len(df) > 2:
                # Predict future hydration trends using linear regression
                model = LinearRegression()
                df['day_of_year'] = df['Date'].apply(lambda x: x.timetuple().tm_yday)
                X = df[['day_of_year']].values
                y = df['Water Intake (ml)'].values
                model.fit(X, y)

                future_days = [datetime.now() + timedelta(days=i) for i in range(1, 8)]
                future_days_num = [d.timetuple().tm_yday for d in future_days]
                future_predictions = model.predict(np.array(future_days_num).reshape(-1, 1))

                future_df = pd.DataFrame({
                    "Date": future_days,
                    "Predicted Intake (ml)": future_predictions
                })
                
                st.subheader("📅 Predicted Water Intake for the Next 7 Days")
                st.dataframe(future_df)
                st.line_chart(future_df, x="Date", y="Predicted Intake (ml)")
                
            else:
                st.warning("⚠️ Not enough data for predictions. Please log more intake.")
        else:
            st.warning("⚠️ No water intake data found. Please log your intake first.")
    
    # 📲 Smart Notifications & Reminders
    st.subheader("🔔 Smart Hydration Reminders")
    reminders = [
        "Time to hydrate! 💦",
        "Don't forget to drink water! 💧",
        "Hydration is key to feeling good! 🌱"
    ]
    st.text(f"🤖 **AI Assistant Reminder:** {random.choice(reminders)}")
    
    # Motivational Message Based on Trends
    if len(df) > 0:
        avg_intake = np.mean(df["Water Intake (ml)"])
        if avg_intake >= hydration_goal:
            st.success("🎉 You're meeting your hydration goal! Keep it up!")
        else:
            st.warning(f"⚠️ You're behind on your hydration goal. Drink more water!")

