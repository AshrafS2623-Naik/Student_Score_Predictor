
st.set_page_config(page_title="EduVision AI", layout="wide")

# ---------------- LOAD ----------------
model = joblib.load("student_model.pkl")
scaler = joblib.load("scaler.pkl")

df = pd.read_csv("student_data.csv")

if "history" not in st.session_state:
    st.session_state.history = []

# ---------------- SIDEBAR ----------------
st.sidebar.title("🎓 EduVision AI")
page = st.sidebar.radio(
    "Navigation",
    ["Home", "Predict", "Analytics", "History"]
)

# ===================================================
# HOME
# ===================================================
if page == "Home":

    st.title("🎓 EduVision AI – Student Score Predictor")

    st.write(
        "AI-powered academic performance predictor that estimates final exam "
        "scores using study habits and lifestyle factors."
    )

    st.markdown("---")

    col1, col2, col3 = st.columns(3)

    with col1:
        st.info("📚 Study Behavior Analysis")

    with col2:
        st.info("🤖 Machine Learning Regression Model")

    with col3:
        st.info("📈 Performance Insights & Trends")

# ===================================================
# PREDICT
# ===================================================
elif page == "Predict":

    st.title("🔮 Predict Final Score")

    col1, col2 = st.columns(2)

    with col1:
        study_hours = st.number_input("Study Hours per Day", 0.5, 12.0, 4.0)
        attendance = st.number_input("Attendance (%)", 40, 100, 80)
        internal_score = st.number_input("Internal Score", 0, 30, 20)

    with col2:
        sleep_hours = st.number_input("Sleep Hours", 3.0, 10.0, 7.0)
        screen_time = st.number_input("Screen Time (hrs)", 0.5, 10.0, 3.0)
        practice_tests = st.number_input("Practice Tests Completed", 0, 50, 10)

    if st.button("Predict Score 🚀"):

        input_data = np.array([[

            study_hours,
            attendance,
            internal_score,
            sleep_hours,
            screen_time,
            practice_tests

        ]])

        input_scaled = scaler.transform(input_data)
        prediction = model.predict(input_scaled)[0]

        st.session_state.history.append(prediction)

        st.success(f"Estimated Final Score: {prediction:.2f}")

# ===================================================
# ANALYTICS
# ===================================================
elif page == "Analytics":

    st.title("📊 Dataset Analytics")

    fig1 = px.histogram(df, x="final_score", nbins=30,
                        title="Final Score Distribution")
    st.plotly_chart(fig1, use_container_width=True)

    fig2 = px.scatter(df, x="study_hours", y="final_score",
                      title="Study Hours vs Final Score")
    st.plotly_chart(fig2, use_container_width=True)

# ===================================================
# HISTORY
# ===================================================
elif page == "History":

    st.title("📜 Prediction History")

    if len(st.session_state.history) == 0:
        st.info("No predictions yet.")
    else:
        history_df = pd.DataFrame({
            "Predicted Scores": st.session_state.history
        })
        st.dataframe(history_df)