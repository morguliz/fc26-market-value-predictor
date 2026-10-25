import pandas as pd

from sklearn.model_selection import train_test_split
from sklearn.linear_model import LinearRegression
from sklearn.metrics import mean_absolute_error

# load dataset
top_250 = pd.read_csv("top_250_players_fc26.csv")


# select features and target
X = top_250[["age", "overall_rating", "potential"]]
y = top_250["value_eur"]


# split data into training and testing sets
X_train, X_test, y_train, y_test = train_test_split(
    X, y,
    random_state=42,
    test_size= 0.2
)

# train linear regression model
model = LinearRegression()
model.fit(X_train, y_train)

# generate predictions
predictions = model.predict(X_test)

# model evaluation
mae = mean_absolute_error(y_test, predictions)

print(f"Mean Absolute Error: €{mae:,.2f}")