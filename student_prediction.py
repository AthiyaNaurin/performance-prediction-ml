import pandas as pd

data = {
    'Study_Hours': [1, 2, 3, 4, 5, 6, 7, 8],
    'Attendance': [60, 65, 70, 75, 80, 85, 90, 95],
    'Previous_Score': [50, 52, 55, 58, 60, 65, 70, 75],
    'Final_Score': [55, 58, 60, 65, 68, 72, 78, 85]
}

df = pd.DataFrame(data)
print(df)
from sklearn.model_selection import train_test_split
from sklearn.linear_model import LinearRegression

X = df[['Study_Hours', 'Attendance', 'Previous_Score']]
y = df['Final_Score']

X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2)

model = LinearRegression()
model.fit(X_train, y_train)
accuracy = model.score(X_test, y_test)
print("Model Accuracy:", accuracy)
new_data = pd.DataFrame([[6, 85, 67]], columns=['Study_Hours', 'Attendance', 'Previous_Score'])

prediction = model.predict(new_data)
print("Predicted Final Score:", prediction[0])


import matplotlib.pyplot as plt

plt.scatter(df['Study_Hours'], df['Final_Score'])
plt.xlabel("Study Hours")
plt.ylabel("Final Score")
plt.title("Study Hours vs Final Score")
plt.show()
