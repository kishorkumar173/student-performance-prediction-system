from fastapi import FastAPI
import joblib
import pandas as pd

app = FastAPI()

model = joblib.load("models/student_model.pkl")

@app.get("/")
def home():
    return {"message": "API is running 🚀"}

@app.post("/predict")
def predict(data: dict):
    try:
        df = pd.DataFrame([data])

        # ✅ FIX COLUMN ORDER (VERY IMPORTANT)
        df = df[[
            "studytime",
            "failures",
            "absences",
            "G1",
            "G2",
            "attendance_pct",
            "study_hours",
            "engagement_score"
        ]]

        pred = model.predict(df)[0]

        return {
            "prediction": int(pred),
            "result": "PASS" if pred == 1 else "FAIL"
        }

    except Exception as e:
        return {"error": str(e)}