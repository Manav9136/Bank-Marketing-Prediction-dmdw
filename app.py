import streamlit as st
import pandas as pd

from sklearn.model_selection import train_test_split
from sklearn.compose import ColumnTransformer
from sklearn.preprocessing import OneHotEncoder, StandardScaler
from sklearn.pipeline import Pipeline
from sklearn.impute import SimpleImputer
from sklearn.tree import DecisionTreeClassifier
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import accuracy_score, f1_score

st.set_page_config(
    page_title="Bank Marketing Predictor",
    page_icon="🏦",
    layout="wide"
)

st.markdown(
    """
    <div style="text-align: center;">
        <h1>🏦 Bank Marketing Campaign Predictor</h1>
        <p>Predict whether a customer is likely to subscribe to a bank term deposit using Data Mining.</p>
    </div>
    """,
    unsafe_allow_html=True
)

@st.cache_data
def load_data():
    return pd.read_csv("bank.csv").drop_duplicates()

data = load_data()

X = data.drop(columns=["deposit", "duration"])
y = data["deposit"].map({"no": 0, "yes": 1})

numeric_cols = X.select_dtypes(include="number").columns.tolist()
categorical_cols = X.select_dtypes(
    include=["object", "category", "bool"]
).columns.tolist()

numeric_pipeline = Pipeline([
    ("imputer", SimpleImputer(strategy="median")),
    ("scaler", StandardScaler())
])

categorical_pipeline = Pipeline([
    ("imputer", SimpleImputer(strategy="most_frequent")),
    ("encoder", OneHotEncoder(handle_unknown="ignore"))
])

preprocessor = ColumnTransformer([
    ("numeric", numeric_pipeline, numeric_cols),
    ("categorical", categorical_pipeline, categorical_cols)
])

X_train, X_test, y_train, y_test = train_test_split(
    X, y,
    test_size=0.20,
    random_state=42,
    stratify=y
)

@st.cache_resource
def train_models(X_train, y_train):
    dt = Pipeline([
        ("preprocessor", preprocessor),
        ("classifier", DecisionTreeClassifier(
            max_depth=6,
            class_weight="balanced",
            random_state=42
        ))
    ])

    lr = Pipeline([
        ("preprocessor", preprocessor),
        ("classifier", LogisticRegression(
            max_iter=2000,
            class_weight="balanced",
            random_state=42
        ))
    ])

    dt.fit(X_train, y_train)
    lr.fit(X_train, y_train)

    return dt, lr

decision_tree, logistic_model = train_models(X_train, y_train)

st.subheader("Enter customer details")

with st.form("customer_form"):
    col1, col2, col3 = st.columns(3)

    with col1:
        age = st.number_input("Age", 18, 100, 35)
        job = st.selectbox("Job", sorted(data["job"].dropna().unique()))
        marital = st.selectbox(
            "Marital status",
            sorted(data["marital"].dropna().unique())
        )
        education = st.selectbox(
            "Education",
            sorted(data["education"].dropna().unique())
        )
        balance = st.number_input(
            "Account balance",
            value=1500
        )

    with col2:
        default = st.selectbox(
            "Credit in default",
            sorted(data["default"].dropna().unique())
        )
        housing = st.selectbox(
            "Housing loan",
            sorted(data["housing"].dropna().unique())
        )
        loan = st.selectbox(
            "Personal loan",
            sorted(data["loan"].dropna().unique())
        )
        contact = st.selectbox(
            "Contact type",
            sorted(data["contact"].dropna().unique())
        )
        months = ["jan", "feb", "mar", "apr", "may", "jun",
                "jul", "aug", "sep", "oct", "nov", "dec"]

        month = st.selectbox(
            "Contact month",
            sorted(data["month"].dropna().unique(), key=months.index)
        )

    with col3:
        day = st.slider("Date of month", 1, 31, 15)
        campaign = st.number_input(
            "Contacts in campaign",
            min_value=1,
            value=2
        )
        pdays = st.number_input(
            "Days since previous contact (-1 = not previously contacted)",
            min_value=-1,
            value=-1
        )
        previous = st.number_input(
            "Previous contacts",
            min_value=0,
            value=0
        )
        poutcome = st.selectbox(
            "Previous campaign outcome",
            sorted(data["poutcome"].dropna().unique())
        )

    model_name = st.selectbox(
        "Choose classification algorithm",
        ["Decision Tree", "Logistic Regression"]
    )

    submitted = st.form_submit_button("Predict customer response")

if submitted:
    customer = pd.DataFrame([{
        "age": age,
        "job": job,
        "marital": marital,
        "education": education,
        "default": default,
        "balance": balance,
        "housing": housing,
        "loan": loan,
        "contact": contact,
        "day": day,
        "month": month,
        "campaign": campaign,
        "pdays": pdays,
        "previous": previous,
        "poutcome": poutcome
    }])[X.columns]

    model = (
        decision_tree
        if model_name == "Decision Tree"
        else logistic_model
    )

    prediction = int(model.predict(customer)[0])
    probability = float(model.predict_proba(customer)[0][1])

    st.markdown(
    f"""
    <div style="text-align: center;">
        <h2>Prediction Result</h2>
        <p>Selected algorithm: <b>{model_name}</b></p>
        <p>Estimated subscription probability</p>
        <h1>{probability:.1%}</h1>
        <div style="
            background-color: {'#173d2c' if prediction == 1 else '#3d2929'};
            color: {'#4ade80' if prediction == 1 else '#fca5a5'};
            padding: 18px;
            border-radius: 10px;
            margin-top: 20px;
        ">
            <b>Model prediction:</b>
            {'Likely to subscribe' if prediction == 1 else 'Unlikely to subscribe'}
        </div>
    </div>
    """,
    unsafe_allow_html=True
)
    st.caption(
        "*This is a statistical prediction based on historical data, "
        "not a guarantee of customer behavior."
    )

st.divider()
