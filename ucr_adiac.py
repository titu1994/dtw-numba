from utils.data_loader import load_dataset
from odtw import KnnDTW

X_train, y_train, X_test, y_test = load_dataset('adiac', normalize_timeseries=True)
print()

# parameters
num_neighbours = 1

model = KnnDTW(num_neighbours)

# fit to the dataset
model.fit(X_train, y_train)

# Predict / Evaluate the score
accuracy = model.evaluate(X_test, y_test)
error = 1. - accuracy

print("*" * 20, "\n")
print("Test Accuracy :", accuracy)
print("Test Error :", error)