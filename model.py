import pandas as pd
from sklearn.model_selection import train_test_split
from sklearn.linear_model import LinearRegression

data = pd.read_csv("student_marks.csv")

X = data[["study_hours", "attendance", "sleep_hours", "previous_marks"]]

y = data["marks"]

X_train, X_test, y_train, y_test = train_test_split(
    X, y, test_size=0.2, random_state=42
)

model = LinearRegression()
model.fit(X_train, y_train)

prediction = model.predict([[6, 80, 7, 70]])

print("Predicted Marks:", prediction[0])

import matplotlib.pyplot as plt

plt.scatter(data["study_hours"], data["marks"])
plt.xlabel("Study Hours")
plt.ylabel("Marks")
plt.title("Study Hours vs Marks")
plt.show()