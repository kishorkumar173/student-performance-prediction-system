import pandas as pd
import numpy as np

np.random.seed(42)

n = 10000  # number of students

# Base (UCI-like features)
data = pd.DataFrame({
    "studytime": np.random.randint(1, 5, n),
    "failures": np.random.randint(0, 4, n),
    "absences": np.random.randint(0, 30, n),
    "G1": np.random.randint(0, 20, n),
    "G2": np.random.randint(0, 20, n),
})

# 🎯 Attendance (inverse of absences)
data["attendance_pct"] = 100 - (data["absences"] * np.random.uniform(1.5, 3.0, n))
data["attendance_pct"] = data["attendance_pct"].clip(40, 100)

# 🎯 Study hours per week
data["study_hours"] = data["studytime"] * np.random.randint(2, 5, n)

# 🎯 Engagement score
data["engagement_score"] = (
    data["studytime"] * 2 +
    (data["attendance_pct"] / 10) +
    np.random.randint(1, 10, n)
)

# 🎯 Final Score (realistic weighted formula)
data["final_score"] = (
    data["G1"] * 0.3 +
    data["G2"] * 0.3 +
    data["attendance_pct"] * 0.2 +
    data["study_hours"] * 0.1 +
    data["engagement_score"] * 0.1
)

# Normalize score to 100
data["final_score"] = (data["final_score"] / data["final_score"].max()) * 100

# 🔥 ADD NOISE (important for realism)
data["final_score"] += np.random.normal(0, 8, n)

# Clip again after noise
data["final_score"] = data["final_score"].clip(0, 100)

# 🔥 DYNAMIC THRESHOLD (balances dataset)
threshold = data["final_score"].quantile(0.6)

data["pass"] = (data["final_score"] >= threshold).astype(int)

# 🎯 Grade Bands
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

# Save dataset
data.to_csv("data/student_performance_final.csv", index=False)

print("✅ Dataset created successfully!")
print(data.head())

print("\nClass Distribution:")
print(data["pass"].value_counts())