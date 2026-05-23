import streamlit as st
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.linear_model import LinearRegression
from sklearn.model_selection import train_test_split
from sklearn.metrics import mean_squared_error, r2_score

# Page setup
st.set_page_config(page_title = "Housing Price Prediction Dashboard", layout="wide")

# Load dataset
df = pd.read_csv("HousingData.csv")
df = df.dropna(subset=["RM","LSTAT","PTRATIO","MEDV","CRIM","NOX","DIS","ZN","RAD","TAX"])

# Features & target
X = df[["RM","LSTAT","PTRATIO","CRIM","NOX","DIS","ZN","RAD","TAX"]]
y = df["MEDV"]

# Train-test split
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size = 0.2, random_state=42)

# Model
model = LinearRegression()
model.fit(X_train, y_train)
y_pred = model.predict(X_test)

# Sidebar menu
st.sidebar.title("📊 Predictive Factors")
option = st.sidebar.radio(
    "Select Analysis:",
    ("Prediction Factors", "Feature Relationships", "Residuals Analysis", "Dataset Overview")
)

# --- Add Model Performance section at bottom of sidebar ---
st.sidebar.markdown("---")  # divider line
st.sidebar.subheader("📊 Model Performance")

# Create two columns inside the sidebar
col1, col2 = st.sidebar.columns(2)

with col1:
    st.markdown(
        f"""
        <div style="background-color:#e6f7ff; padding:10px; border-radius:8px; text-align:center;">
            <h4 style="margin:0;font-weight:bold; color:#007acc;">R² Score</h4>
            <p style="font-size:20px; font-weight:bold; color:#007acc;">{r2_score(y_test, y_pred):.2f}</p>
        </div>
        """,
        unsafe_allow_html=True
    )

with col2:
    st.markdown(
        f"""
        <div style="background-color:#fff0f0; padding:10px; border-radius:8px; text-align:center;">
            <h4 style="margin:0;font-weight:bold; color:#cc0000;">MSE</h4>
            <p style="font-size:20px; font-weight:bold; color:#cc0000;">{mean_squared_error(y_test, y_pred):.2f}</p>
        </div>
        """,
        unsafe_allow_html=True
    )

# Explanations below the cards
st.sidebar.caption("**R² Score:** Measures how well the model explains variation in house prices. Closer to 1 = better fit.")
st.sidebar.caption("**MSE:** Mean Squared Error shows average squared difference between predicted and actual prices. Lower = more accurate.")


# --- Prediction Tool (two-column fixed layout) ---
if option == "Prediction Factors":
    st.title("🏠 House Price Prediction")
    col1, col2 = st.columns([1,2])

    with col1:
        st.subheader("🔧 Select Features")
        rooms = st.slider("Average number of rooms (RM)", float(X["RM"].min()), float(X["RM"].max()), 6.0)
        lstat = st.slider("% Lower Status Population (LSTAT)", float(X["LSTAT"].min()), float(X["LSTAT"].max()), 12.0)
        ptratio = st.slider("Pupil-Teacher Ratio (PTRATIO)", float(X["PTRATIO"].min()), float(X["PTRATIO"].max()), 18.0)
        crim = st.slider("Crime Rate (CRIM)", float(X["CRIM"].min()), float(X["CRIM"].max()), 0.1)
        nox = st.slider("Air Pollution (NOX)", float(X["NOX"].min()), float(X["NOX"].max()), 0.5)
        dis = st.slider("Distance to Employment Centres (DIS)", float(X["DIS"].min()), float(X["DIS"].max()), 5.0)
        zn = st.slider("Land Zoned for Commercial (ZN)", float(X["ZN"].min()), float(X["ZN"].max()), 10.0)
        rad = st.slider("Highway Accessibility Index (RAD)", float(X["RAD"].min()), float(X["RAD"].max()), 5.0)
        tax = st.slider("Property Tax Rate (TAX)", float(X["TAX"].min()), float(X["TAX"].max()), 300.0)

    with col2:
        st.subheader("📖 Sentiment Analysis of Factors")
        predicted_price = model.predict([[rooms, lstat, ptratio, crim, nox, dis, zn, rad, tax]])[0]

        st.markdown(f"""
        **Feature Choices:**
        - 🟢 Rooms: {rooms} → More rooms generally increase house value.
        - 🔴 LSTAT: {lstat}% → Higher socio-economic disadvantage lowers house value.
        - 🔴 PTRATIO: {ptratio} → Higher ratios (crowded schools) reduce house value.
        - 🔴 CRIM: {crim} → Higher crime rate reduces house value.
        - 🔴 NOX: {nox} → Higher air pollution reduces house value.
        - 🟢 DIS: {dis} → Greater distance to employment centres may increase suburban appeal.
        - 🟢 ZN: {zn} → More commercial zoning can raise property values.
        - 🟢 RAD: {rad} → Higher highway accessibility index can increase value.
        - 🔴 TAX: {tax} → Higher property tax rate may reduce affordability.

        **Prediction Result:**
        - 💰 Estimated House Price: **${predicted_price*1000:.2f}**
        """)

        st.info(f"Example: A house with {rooms} rooms, crime rate {crim}, NOX {nox}, distance {dis}, ZN {zn}, RAD {rad}, TAX {tax}, LSTAT {lstat}% and PTRATIO {ptratio} is predicted to cost around ${predicted_price*1000:.2f}.")

# --- Feature Relationships (top-bottom layout) ---
elif option == "Feature Relationships":
    st.title("📈 Feature vs Price Relationships")
    fig, axes = plt.subplots(1, 3, figsize=(18,5))
    sns.scatterplot(x=df["RM"], y=df["MEDV"], ax=axes[0], color="royalblue")
    axes[0].set_title("Rooms vs Price")
    sns.scatterplot(x=df["LSTAT"], y=df["MEDV"], ax=axes[1], color="darkorange")
    axes[1].set_title("LSTAT vs Price")
    sns.scatterplot(x=df["PTRATIO"], y=df["MEDV"], ax=axes[2], color="seagreen")
    axes[2].set_title("PTRATIO vs Price")
    st.pyplot(fig)
    st.info("**Notes:**\n- More rooms → higher prices.\n- Higher LSTAT → lower prices.\n- Higher PTRATIO → lower prices.")

# --- Residuals Analysis (top-bottom layout, resized plot) ---
elif option == "Residuals Analysis":
    st.title("🔍 Residuals Analysis")
    residuals = y_test - y_pred
    plt.figure(figsize=(12,5))  # resized plot
    sns.scatterplot(x=y_pred, y=residuals, color="crimson")
    plt.axhline(0, color="black", linestyle="--")
    plt.xlabel("Predicted Price")
    plt.ylabel("Residuals")
    plt.title("Residuals vs Predicted")
    st.pyplot(plt)
    st.warning("Residuals show prediction errors. A balanced spread around zero means the model is unbiased. Large deviations highlight weak spots.")

# --- Dataset Overview (top-bottom layout, smaller scroll window) ---
elif option == "Dataset Overview":
    st.title("📂 Dataset Overview")
    st.dataframe(df.head(20), height=200)  # reduced height for scroll window
    st.success("This dataset contains housing features like rooms, socio-economic status, crime rate, air pollution, distance to employment centres, zoning, highway accessibility, and tax rate, used to predict median house values.")