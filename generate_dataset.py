import pandas as pd
import numpy as np

np.random.seed(42)

n = 10000  # number of students

# -----------------------------
# BASE FEATURES
# -----------------------------
data = pd.DataFrame({
    "studytime": np.random.randint(1, 5, n),
    "failures": np.random.randint(0, 4, n),
    "absences": np.random.randint(0, 30, n),
    "G1": np.random.randint(5, 20, n),   # avoid too many zeros
    "G2": np.random.randint(5, 20, n),
})

# -----------------------------
# ATTENDANCE
# -----------------------------
data["attendance_pct"] = 100 - (data["absences"] * np.random.uniform(1.5, 2.5, n))
data["attendance_pct"] = data["attendance_pct"].clip(50, 100)

# -----------------------------
# STUDY HOURS
# -----------------------------
data["study_hours"] = data["studytime"] * np.random.randint(2, 6, n)

# -----------------------------
# ENGAGEMENT SCORE
# -----------------------------
data["engagement_score"] = (
    data["studytime"] * 3 +
    (data["attendance_pct"] / 8) +
    np.random.randint(2, 10, n)
)

# -----------------------------
# FINAL SCORE (STRONG LOGIC)
# -----------------------------
data["final_score"] = (
    data["G1"] * 0.35 +
    data["G2"] * 0.35 +
    data["attendance_pct"] * 0.15 +
    data["study_hours"] * 0.1 +
    data["engagement_score"] * 0.05
)

# Normalize to 100
data["final_score"] = (data["final_score"] / data["final_score"].max()) * 100

# -----------------------------
# ADD REALISTIC NOISE
# -----------------------------
data["final_score"] += np.random.normal(0, 5, n)
data["final_score"] = data["final_score"].clip(0, 100)

# -----------------------------
# ADD FAILURE PENALTY (IMPORTANT)
# -----------------------------
data["final_score"] -= data["failures"] * 5

# -----------------------------
# TARGET (BALANCED)
# -----------------------------
threshold = data["final_score"].median()   # better than quantile(0.6)
data["pass"] = (data["final_score"] >= threshold).astype(int)

# -----------------------------
# GRADE SYSTEM
# -----------------------------
def grade(score):
    if score >= 80:
        return "A"
    elif score >= 65:
        return "B"
    elif score >= 50:
        return "C"
    else:
        return "F"

data["grade"] = data["final_score"].apply(grade)

# -----------------------------
# SAVE DATASET
# -----------------------------
data.to_csv("data/student_performance_final.csv", index=False)

print("✅ Dataset created successfully!")
print(data.head())

print("\nClass Distribution:")
print(data["pass"].value_counts())