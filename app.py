"""Streamlit app: predict whether a LendingClub loan will charge off.

Loads the artifacts exported by LoanDefaultProbability.ipynb (section 7.3) and
replays the same preprocessing used during training: label-encode categoricals,
standard-scale, PCA-transform, then run the tuned model.
"""

import numpy as np
import pandas as pd
import streamlit as st
import joblib

st.set_page_config(page_title="Loan Default Predictor", page_icon="💰", layout="centered")

ARTIFACTS = {
    "model": "best_model.joblib",
    "encoders": "label_encoders.joblib",
    "scaler": "pca_scaler.joblib",
    "pca": "pca_transform.joblib",
}


@st.cache_resource
def load_artifacts():
    missing = [path for path in ARTIFACTS.values() if not __import__("os").path.exists(path)]
    if missing:
        raise FileNotFoundError(missing)
    return {name: joblib.load(path) for name, path in ARTIFACTS.items()}


try:
    artifacts = load_artifacts()
except FileNotFoundError as e:
    st.error(
        "Missing model artifact(s): "
        + ", ".join(e.args[0])
        + ".\n\nRun `LoanDefaultProbability.ipynb` end-to-end first "
        "(section 7.3 saves these files)."
    )
    st.stop()

model = artifacts["model"]
encoders = artifacts["encoders"]
scaler = artifacts["scaler"]
pca = artifacts["pca"]

FEATURE_COLUMNS = list(scaler.feature_names_in_)
CATEGORICAL_COLUMNS = list(encoders.keys())

st.title("💰 Loan Default Predictor")
st.write(
    "Estimates the probability that a loan will **charge off** (default) rather than "
    "be fully paid, using the tuned model from the accompanying notebook."
)

with st.form("loan_form"):
    st.subheader("Loan Details")
    col1, col2 = st.columns(2)
    with col1:
        loan_amnt = st.number_input(
            "Loan amount ($)", min_value=1000, max_value=40000, value=15000, step=500,
            help="Total amount requested by the borrower. Also used as the funded amount "
            "(assumes the loan is funded in full, as most approved loans are).",
        )
        term = st.selectbox(
            "Term (months)", options=[36, 60],
            help="Number of monthly payments for the loan — 3 years or 5 years.",
        )
        int_rate = st.number_input(
            "Interest rate (%)", min_value=5.0, max_value=31.0, value=13.0, step=0.1,
            help="Annual interest rate assigned to the loan, driven mainly by grade/sub-grade.",
        )
        installment = st.number_input(
            "Monthly installment ($)", min_value=25.0, max_value=1600.0, value=450.0, step=10.0,
            help="Fixed monthly payment the borrower owes once the loan is funded.",
        )
    with col2:
        grade = st.selectbox(
            "Grade", options=encoders["grade"].classes_,
            help="LendingClub's risk grade for the loan (A = lowest risk, G = highest risk); "
            "it's the main driver of the interest rate.",
        )
        sub_grade = st.selectbox(
            "Sub-grade", options=encoders["sub_grade"].classes_,
            help="Finer-grained risk tier within the grade — e.g. C1 is lower risk than C5.",
        )
        purpose = st.selectbox(
            "Purpose", options=encoders["purpose"].classes_,
            help="Borrower's stated reason for taking out the loan.",
        )
        initial_list_status = st.selectbox(
            "Initial list status", options=encoders["initial_list_status"].classes_,
            help="Whether the loan was initially listed as a whole loan ('w') or "
            "a fractional loan ('f') on the LendingClub platform.",
        )

    st.subheader("Borrower Profile")
    col3, col4 = st.columns(2)
    with col3:
        annual_inc = st.number_input(
            "Annual income ($)", min_value=4000, max_value=2_000_000, value=65000, step=1000,
            help="Borrower's self-reported annual income. The model actually trains on "
            "log10(income), which flattens the effect of very high incomes.",
        )
        home_ownership = st.selectbox(
            "Home ownership", options=encoders["home_ownership"].classes_,
            help="Home ownership status the borrower provided during registration.",
        )
        verification_status = st.selectbox(
            "Verification status", options=encoders["verification_status"].classes_,
            help="Whether LendingClub verified the borrower's reported income.",
        )
        application_type = st.selectbox(
            "Application type", options=encoders["application_type"].classes_,
            help="Whether this is an individual or joint loan application.",
        )
    with col4:
        addr_state = st.selectbox(
            "State", options=encoders["addr_state"].classes_,
            help="U.S. state the borrower provided in the loan application.",
        )
        dti = st.number_input(
            "Debt-to-income ratio (%)", min_value=0.0, max_value=60.0, value=18.0, step=0.5,
            help="Total monthly debt payments (excluding mortgage), divided by monthly income.",
        )
        fico_score = st.number_input(
            "FICO score", min_value=300, max_value=850, value=690, step=1,
            help="Borrower's FICO credit score (300-850; higher is better credit). The model "
            "uses the average of the borrower's reported low/high FICO range.",
        )
        earliest_cr_line = st.number_input(
            "Earliest credit line (year)", min_value=1950, max_value=2017, value=2005, step=1,
            help="Year the borrower's earliest reported credit line was opened — "
            "a proxy for how long they've had credit history.",
        )

    st.subheader("Credit History")
    col5, col6 = st.columns(2)
    with col5:
        open_acc = st.number_input(
            "Open accounts", min_value=0, max_value=100, value=11, step=1,
            help="Number of currently open credit lines in the borrower's credit file.",
        )
        revol_util = st.number_input(
            "Revolving utilization (%)", min_value=0.0, max_value=200.0, value=55.0, step=1.0,
            help="Revolving line utilization rate: how much of their available revolving "
            "credit (e.g. credit cards) the borrower is currently using.",
        )
        acc_open_past_24mths = st.number_input(
            "Accounts opened (past 24 months)", min_value=0, max_value=50, value=4, step=1,
            help="Number of credit accounts the borrower opened in the last 2 years.",
        )
        avg_cur_bal = st.number_input(
            "Average current balance ($)", min_value=0, max_value=500000, value=13000, step=500,
            help="Average current balance across all of the borrower's open accounts.",
        )
        mort_acc = st.number_input(
            "Mortgage accounts", min_value=0, max_value=40, value=1, step=1,
            help="Number of mortgage accounts the borrower currently has.",
        )
    with col6:
        bc_open_to_buy = st.number_input(
            "Bankcard open-to-buy ($)", min_value=0, max_value=250000, value=9000, step=500,
            help="Total remaining spending capacity across the borrower's bankcards "
            "(credit limit minus current balance).",
        )
        bc_util = st.number_input(
            "Bankcard utilization (%)", min_value=0.0, max_value=250.0, value=60.0, step=1.0,
            help="Ratio of current bankcard balances to bankcard credit limits.",
        )
        mo_sin_old_rev_tl_op = st.number_input(
            "Months since oldest revolving account opened", min_value=0, max_value=800, value=180, step=1,
            help="Age, in months, of the borrower's oldest revolving (e.g. credit card) account.",
        )
        mo_sin_rcnt_rev_tl_op = st.number_input(
            "Months since most recent revolving account opened", min_value=0, max_value=400, value=13, step=1,
            help="Months since the borrower's most recently opened revolving account — "
            "a small number can indicate recent credit-seeking behavior.",
        )
        num_actv_rev_tl = st.number_input(
            "Active revolving accounts", min_value=0, max_value=50, value=6, step=1,
            help="Number of currently active revolving trade lines (credit cards, lines of credit, etc.).",
        )

    submitted = st.form_submit_button("Predict")

if submitted:
    raw = {
        "loan_amnt": loan_amnt,
        "funded_amnt": loan_amnt,
        "term": term,
        "int_rate": int_rate,
        "installment": installment,
        "grade": grade,
        "sub_grade": sub_grade,
        "home_ownership": home_ownership,
        "verification_status": verification_status,
        "purpose": purpose,
        "addr_state": addr_state,
        "dti": dti,
        "earliest_cr_line": earliest_cr_line,
        "open_acc": open_acc,
        "revol_util": revol_util,
        "initial_list_status": initial_list_status,
        "application_type": application_type,
        "acc_open_past_24mths": acc_open_past_24mths,
        "avg_cur_bal": avg_cur_bal,
        "bc_open_to_buy": bc_open_to_buy,
        "bc_util": bc_util,
        "mo_sin_old_rev_tl_op": mo_sin_old_rev_tl_op,
        "mo_sin_rcnt_rev_tl_op": mo_sin_rcnt_rev_tl_op,
        "mort_acc": mort_acc,
        "num_actv_rev_tl": num_actv_rev_tl,
        "log_annual_inc": np.log10(annual_inc + 1),
        "fico_score": fico_score,
    }

    row = pd.DataFrame([raw])
    for col in CATEGORICAL_COLUMNS:
        row[col] = encoders[col].transform(row[col].astype(str))
    row = row[FEATURE_COLUMNS]

    scaled = scaler.transform(row)
    components = pca.transform(scaled)
    components_df = pd.DataFrame(components, columns=model.feature_names_in_)

    charge_off_prob = model.predict_proba(components_df)[0, 1]
    prediction = model.predict(components_df)[0]

    st.subheader("Result")
    if prediction == 1:
        st.error(f"⚠️ Predicted: **Charge-Off risk** — {charge_off_prob:.1%} probability of default")
    else:
        st.success(f"✅ Predicted: **Likely to be fully paid** — {charge_off_prob:.1%} probability of default")
    st.progress(min(max(charge_off_prob, 0.0), 1.0))
    st.caption(
        "Probability is the model's estimated chance of charge-off. "
        "This is a demo model trained on a sampled, balanced dataset — not financial advice."
    )
