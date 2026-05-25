from flask import Flask, render_template, request
import csv
import pandas as pd
from sklearn.linear_model import LinearRegression

app = Flask(__name__)

data = pd.read_csv("student_marks.csv")

X = data[["study_hours", "attendance", "sleep_hours", "previous_marks"]]
y = data["marks"]

model = LinearRegression()
model.fit(X, y)

@app.route("/", methods=["GET", "POST"])
def home():

    prediction = None
    study_hours = 0
    attendance = 0
    sleep_hours = 0
    previous_marks = 0

    if request.method == "POST":

        study_hours = float(request.form["study_hours"])
        attendance = float(request.form["attendance"])
        sleep_hours = float(request.form["sleep_hours"])
        previous_marks = float(request.form["previous_marks"])

        result = model.predict([[
            study_hours,
            attendance,
            sleep_hours,
            previous_marks
        ]])

        prediction = round(max(0, min(100, result[0])), 2)

        with open("user_data.csv", "a", newline="") as file:
            writer = csv.writer(file)

            writer.writerow([
                study_hours,
                attendance,
                sleep_hours,
                previous_marks,
                prediction
            ])

    return render_template(
        "index.html",
        prediction=prediction,
        study_hours=study_hours,
        attendance=attendance,
        sleep_hours=sleep_hours,
        previous_marks=previous_marks
    )

if __name__ == "__main__":
    app.run(debug=True)