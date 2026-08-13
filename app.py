import streamlit as st
import joblib
import numpy as np
import pandas as pd

model = joblib.load("fraud_detection_pipeline.pkl")

st.title("Welcome to the Fraudulent Transactions Detector App")
st.divider()

st.markdown("""Predict whether a transaction  is **FAIR or FRAUD** using a **XGBoost Classifier Model** trained on the Transactions dataset with almost **6 Million rows**.
**Model Information**
- 🌲 Algorithm: XGBoost Classifier 
- 🎯 Test Accuracy: **99.7%** with Recall: **0.98**. Signalling almost all fraudulent transactions detected.
- 📊 Evaluation: Accuracy, Confusion Matrix & Classification Report""")

st.divider()

st.write(
    "Please select the **appropriate** values and click **Predict** to check weather transaction is Fraud or Not."
)

st.divider()

type = st.selectbox("Transaction Type", ["TRANSFER","CASH_OUT","DEBIT","CASH_IN","PAYMENT",])

amount= st.number_input("Enter Amount", min_value=0, max_value=1000000, value=10000)

oldbalanceOrg= st.number_input("Enter Old Balance of Sender", min_value=0, max_value=1000000, value=10000)

newbalanceOrig= st.number_input("Enter New Balance of Sender", min_value=0, max_value=1000000, value=10000)

oldbalanceDest= st.number_input("Enter Old Balance of Reciever", min_value=0, max_value=1000000, value=10000)

newbalanceDest= st.number_input("Enter New Balance of Reciever", min_value=0, max_value=1000000, value=10000)

BalanceDiffOrig = oldbalanceOrg - newbalanceOrig

errorBalanceDest = round(amount - (newbalanceDest - oldbalanceDest), 2)

errorBalanceOrig = round(amount - (oldbalanceOrg - newbalanceOrig), 2)

is_dest_zero_after_transaction = int((oldbalanceDest == 0) and (newbalanceDest == 0) and (amount > 0))

st.divider()

predict_button = st.button("Predict")
if predict_button:
    input_df = pd.DataFrame(
        [
            {
                "type": type,
                "amount": amount,
                "oldbalanceOrg": oldbalanceOrg,
                "newbalanceOrig": newbalanceOrig,
                "oldbalanceDest": oldbalanceDest,
                "newbalanceDest": newbalanceDest,
                "BalanceDiffOrig": BalanceDiffOrig,
                "errorBalanceDest": errorBalanceDest,
                "errorBalanceOrig": errorBalanceOrig,
                "is_dest_zero_after_transaction0" : is_dest_zero_after_transaction,
            }
        ])
    prediction = model.predict(input_df)
    probability = model.predict_proba(input_df)

    if prediction[0] == 1:
        st.error("🚨 **FRAUDULENT TRANSACTION**")
    else:
        st.success("✅ **FAIR TRANSACTION**")

    st.write(f"**Probability of Legitimate Transaction:** {probability[0][0]*100:.2f}%")
    st.write(f"**Probability of Fraud Risk:** {probability[0][1]*100:.2f}%")

else:
    st.info("Enter the transaction details above and click **Predict**.")