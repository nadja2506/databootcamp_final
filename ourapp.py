import streamlit as st
import pandas as pd
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler
from sklearn.linear_model import LogisticRegression
from sklearn.neighbors import KNeighborsClassifier
from sklearn.ensemble import RandomForestClassifier, GradientBoostingClassifier
from sklearn.metrics import accuracy_score, recall_score, precision_score
#Title
st.title("Up/Down Day Predictor")


# upload the raw CVs 

fab_file = st.file_uploader("Upload FAB.csv", type=["csv"])
taqa_file = st.file_uploader("Upload TAQA.csv", type=["csv"])

if fab_file is None or taqa_file is None:
  st.info("Please upload both FAB.csv and TAQA.csv to continue.")
  st.stop()

fab_raw = pd.read_csv(fab_file)
taqa_raw = pd.read_csv(taqa_file)

st.write("FAB preview:")
st.write(fab_raw.head())
st.write("TAQA preview:")
st.write(taqa_raw.head())

# preprocessing 

def prepuae(df, company, sector):
  df = df.copy()

  df = df.rename(columns={"Price": "AdjClose"})

  df["Date"] = pd.to_datetime(df["Date"], errors="coerce")

  df["AdjClose"] = pd.to_numeric(df["AdjClose"], errors="coerce")

  #drop NAs
  df = df.dropna(subset=["Date", "AdjClose"])
  df = df.sort_values("Date")

  df["Return"] = df["AdjClose"].pct_change()

  df["Company"] = company
  df["Market"] = "ADX"
  df["Sector"] = sector

  return df[["Date", "Company", "Market", "Sector", "AdjClose", "Return"]]
fab = prepuae(fab_raw, "FAB", "Banking")
taqa = prepuae(taqa_raw, "TAQA", "Energy")

# drop NA returns
fab = fab.dropna(subset=["Return"])
taqa = taqa.dropna(subset=["Return"])

# combine both companies 
full = pd.concat([fab, taqa], ignore_index=True)

full = full.sort_values(["Company", "Date"]).reset_index(drop=True)

full["UpDay"] = (full["Return"] > 0).astype(int)

full["Return_lag1"] = full.groupby("Company")["Return"].shift(1)
full["Return_lag2"] = full.groupby("Company")["Return"].shift(2)
full["Return_lag3"] = full.groupby("Company")["Return"].shift(3)

full["RollingVol_5"] = (
  full.groupby("Company")["Return"]
  .rolling(window=5)
  .std()
  .reset_index(level=0, drop=True)
)

full["RollingMean_5"] = (
  full.groupby("Company")["Return"]
  .rolling(window=5)
  .mean()
  .reset_index(level=0, drop=True)
)

full= full.dropna().reset_index(drop=True)

st.subheader("Engineered data preview")
st.write(full.head())

# pick company to model 

company_choice = st.selectbox("Choose company", sorted(full["Company"].unique()))
data = full[full["Company"] == company_choice].copy()

# features and target
X = data[["Return_lag1", "Return_lag2", "Return_lag3", "RollingVol_5", "RollingMean_5" ]]
y = data["UpDay"]

# model 

model_name = st.selectbox(
  "Choose a model",
  ["Logistic Regression", "KNN", "Random Forest", "Gradient Boosting"]

)

# time respecting split 
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.3, shuffle=False)



scaler = None
if model_name == "Logistic Regression":
  scaler = StandardScaler()
  X_train_use = scaler.fit_transform(X_train)
  X_test_use = scaler.transform(X_test)
  model = LogisticRegression(max_iter = 1000)
elif model_name == "KNN":
  scaler = StandardScaler()
  X_train_use = scaler.fit_transform(X_train)
  X_test_use = scaler.transform(X_test)
  model = KNeighborsClassifier(n_neighbors=20)
elif model_name == "Random Forest":
  X_train_use, X_test_use = X_train, X_test
  model = RandomForestClassifier(n_estimators=200, max_depth=6, random_state=42)
else:
  X_train_use, X_test_use = X_train, X_test
  model = GradientBoostingClassifier(
    n_estimators = 300,
    learning_rate = 0.05,
    max_depth = 3,
    random_state = 42
  )
#Train button
if st.button("Train and Evaluate"):
  model.fit(X_train_use, y_train)
  preds = model.predict(X_test_use)
  #Results
  st.subheader("Model evaluation:")
  st.write("Accuracy", round(accuracy_score(y_test, preds),3))
  st.write("Recall", round(recall_score(y_test, preds),3))
  st.write("Precision", round(precision_score(y_test, preds),3))

  latest = X.iloc[[-1]]
  if scaler is not None:
    latest = scaler.transform(latest)
  latest_pred = model.predict(latest)[0]
  st.subheader("Latest day prediction")
  st.success("Up(1)" if latest_pred ==1 else "Down(0)")




