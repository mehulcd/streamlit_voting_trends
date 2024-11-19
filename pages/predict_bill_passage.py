# Import libraries
import pandas as pd
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler
from sklearn.linear_model import LogisticRegression
from sklearn.feature_selection import RFE
from sklearn.metrics import accuracy_score, classification_report, roc_auc_score, roc_curve
import matplotlib.pyplot as plt
import numpy as np
import streamlit as st

# Streamlit Title
st.title("Bill Prediction Analysis")

# Upload CSV file
uploaded_file = 'data/flattened_votes.csv'
if uploaded_file:

    df = pd.read_csv(uploaded_file)
    
    df['Bill Passed'] = df['Bill Result'].apply(lambda x: 1 if x in ['Passed', 'Bill Passed'] else 0)

    # Health-related analysis
    df['Is_Health_Related'] = df['Subject'].str.contains(
        'Health|Clinic|Hospital|Prescription|Mental|Healthcare|Vaccination|Pandemic|Covid|Medical', 
        case=False, na=False
    ).astype(int)

    df_encoded = pd.get_dummies(df[['State', 'Party']], drop_first=True)
    df_encoded['Is_Health_Related'] = df['Is_Health_Related']

    X = df_encoded
    y = df['Bill Passed']

    # Logistic Regression Model
    X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)

    log_reg = LogisticRegression(max_iter=1000, solver='saga')
    log_reg.fit(X_train, y_train)

    y_pred = log_reg.predict(X_test)
    y_pred_proba = log_reg.predict_proba(X_test)[:, 1]

    # Display Metrics
    st.subheader("Model Evaluation Metrics")
    st.write(f"Accuracy: {accuracy_score(y_test, y_pred):.2f}")
    st.text("Classification Report")
    st.text(classification_report(y_test, y_pred))

    # Interactive Plots
    st.subheader("Plots")
    plot_type = st.selectbox("Select Plot", [
        "Histogram (Predicted Probabilities)",
        "ROC Curve",
        "Predicted Probabilities vs Actual Outcomes"
    ])

    if plot_type == "Histogram (Predicted Probabilities)":
        fig, ax = plt.subplots(figsize=(10, 6))
        n, bins, patches = ax.hist(y_pred_proba, bins=20, edgecolor='black')
        colors = plt.cm.prism(np.linspace(0, 1, len(patches)))
        for patch, color in zip(patches, colors):
            patch.set_facecolor(color)
        ax.set_xlabel("Predicted Probability of Passing")
        ax.set_ylabel("Frequency")
        ax.set_title("Distribution of Predicted Passing Probabilities")
        st.pyplot(fig)

    elif plot_type == "ROC Curve":
        fpr, tpr, thresholds = roc_curve(y_test, y_pred_proba)
        roc_auc = roc_auc_score(y_test, y_pred_proba)

        fig, ax = plt.subplots(figsize=(10, 6))
        ax.plot(fpr, tpr, color='blue', label=f'ROC Curve (AUC = {roc_auc:.2f})')
        ax.plot([0, 1], [0, 1], color='gray', linestyle='--')
        ax.set_xlabel("False Positive Rate")
        ax.set_ylabel("True Positive Rate")
        ax.set_title("ROC Curve")
        ax.legend()
        st.pyplot(fig)

    elif plot_type == "Predicted Probabilities vs Actual Outcomes":
        fig, ax = plt.subplots(figsize=(10, 6))
        ax.scatter(np.arange(len(y_test)), y_pred_proba, color='blue', label='Predicted Probability')
        ax.scatter(np.arange(len(y_test)), y_test, color='green', alpha=0.3, label='Actual Outcome')
        ax.set_xlabel("Sample Index")
        ax.set_ylabel("Probability of Passing")
        ax.set_title("Predicted Probability vs Actual Outcome")
        ax.legend()
        st.pyplot(fig)

    # Sensitivity Analysis
    st.subheader("Sensitivity Analysis")
    feature = "Is_Health_Related"
    X_test_sensitivity = X_test.copy()
    X_test_sensitivity[feature] = 0
    predicted_prob_0 = log_reg.predict_proba(X_test_sensitivity)[:, 1].mean()

    X_test_sensitivity[feature] = 1
    predicted_prob_1 = log_reg.predict_proba(X_test_sensitivity)[:, 1].mean()

    impact = predicted_prob_1 - predicted_prob_0
    st.write(f"When {feature} = 0, Average Predicted Probability of Passing: {predicted_prob_0:.4f}")
    st.write(f"When {feature} = 1, Average Predicted Probability of Passing: {predicted_prob_1:.4f}")
    st.write(f"Impact of {feature} on Probability of Passing: {impact:.4f}")

    # Coefficients
    coefficients_health = pd.Series(log_reg.coef_[0], index=X_train.columns).sort_values(ascending=False)
    st.subheader("Feature Coefficients")
    st.write("Top Positive Coefficients (Increase Passing Probability):")
    st.write(coefficients_health.head(10))
    st.write("Top Negative Coefficients (Decrease Passing Probability):")
    st.write(coefficients_health.tail(10))
