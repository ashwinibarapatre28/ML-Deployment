from flask import Flask, render_template, request, jsonify
import numpy as np
import pickle
import pandas as pd

app = Flask(__name__)

# Load model
model = pickle.load(open("model.pkl", "rb"))

@app.route("/")
def home():
    return render_template("index.html")

@app.route("/predict", methods=["POST"])
def predict():
    try:
        data = request.json
        age = int(data["age"])
        bmi = float(data["bmi"])
        children = int(data["children"])
        smoker = int(data["smoker"])
        male = int(data["male"])
        region = data.get("region", "northeast")

        region_northwest = 1 if region == 'northwest' else 0
        region_southeast = 1 if region == 'southeast' else 0
        region_southwest = 1 if region == 'southwest' else 0

        # Create DataFrame with exact feature names
        features = pd.DataFrame([{
            'age': age,
            'bmi': bmi,
            'children': children,
            'sex_male': male,
            'smoker_yes': smoker,
            'region_northwest': region_northwest,
            'region_southeast': region_southeast,
            'region_southwest': region_southwest
        }])

        prediction = model.predict(features)[0]

        # Detailed insights based on logic
        insights = []
        if smoker == 1:
            insights.append("🔴 Smoking significantly increases your insurance premium.")
        else:
            insights.append("🟢 Non-smoker status keeps your base premium lower.")
            
        if bmi < 18.5:
            insights.append("⚠️ Your BMI indicates you are underweight.")
        elif 18.5 <= bmi <= 24.9:
            insights.append("🟢 Your BMI is in the healthy range.")
        elif 25 <= bmi <= 29.9:
            insights.append("⚠️ Your BMI indicates you are overweight.")
        else:
            insights.append("🔴 Your BMI is in the obese range, which raises insurance costs.")
            
        if age > 50:
            insights.append("ℹ️ Age over 50 generally incurs higher baseline premiums.")

        return jsonify({
            "success": True,
            "prediction": round(prediction, 2),
            "insights": insights
        })

    except Exception as e:
        return jsonify({"success": False, "error": str(e)})

if __name__ == "__main__":
    app.run(debug=True)