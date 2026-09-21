import csv
import math
import random


DATA_FILE = "data/placement_data.csv"


def sigmoid(z):
    z = max(min(z, 500), -500)
    return 1 / (1 + math.exp(-z))


def load_data():
    X = []
    y = []

    with open(DATA_FILE, "r", newline="", encoding="utf-8") as file:
        reader = csv.DictReader(file)

        for row in reader:
            features = [
                float(row["CGPA"]),
                float(row["Attendance"]),
                float(row["CRT_Score"]),
            ]

            target = 1 if row["Placed"].strip().lower() == "yes" else 0

            X.append(features)
            y.append(target)

    return X, y


def normalize(X):
    means = []
    stds = []

    for column in range(len(X[0])):
        values = [row[column] for row in X]
        mean = sum(values) / len(values)

        variance = sum((value - mean) ** 2 for value in values) / len(values)
        std = math.sqrt(variance)

        if std == 0:
            std = 1

        means.append(mean)
        stds.append(std)

    normalized = []

    for row in X:
        normalized.append([
            (row[i] - means[i]) / stds[i]
            for i in range(len(row))
        ])

    return normalized, means, stds


def train_logistic_regression(X, y, learning_rate=0.1, epochs=3000):
    weights = [0.0] * len(X[0])
    bias = 0.0

    for _ in range(epochs):
        weight_gradients = [0.0] * len(weights)
        bias_gradient = 0.0

        for features, target in zip(X, y):
            score = bias + sum(
                weight * feature
                for weight, feature in zip(weights, features)
            )

            prediction = sigmoid(score)
            error = prediction - target

            for i in range(len(weights)):
                weight_gradients[i] += error * features[i]

            bias_gradient += error

        n = len(X)

        for i in range(len(weights)):
            weights[i] -= learning_rate * weight_gradients[i] / n

        bias -= learning_rate * bias_gradient / n

    return weights, bias


def predict_probability(features, weights, bias, means, stds):
    normalized = [
        (features[i] - means[i]) / stds[i]
        for i in range(len(features))
    ]

    score = bias + sum(
        weights[i] * normalized[i]
        for i in range(len(weights))
    )

    return sigmoid(score)


def accuracy(X, y, weights, bias, means, stds):
    correct = 0

    for features, target in zip(X, y):
        probability = predict_probability(
            features, weights, bias, means, stds
        )

        prediction = 1 if probability >= 0.5 else 0

        if prediction == target:
            correct += 1

    return correct / len(y)


def main():
    print("=" * 55)
    print("PLACEMENT PREDICTION ML")
    print("=" * 55)

    X, y = load_data()

    print(f"\nDataset size: {len(X)} students")
    print("Features: CGPA, Attendance, CRT Score")
    print("Target: Placement (Yes/No)")

    # Shuffle and split data
    combined = list(zip(X, y))
    random.Random(42).shuffle(combined)

    split_index = int(len(combined) * 0.8)

    train_data = combined[:split_index]
    test_data = combined[split_index:]

    X_train = [item[0] for item in train_data]
    y_train = [item[1] for item in train_data]

    X_test = [item[0] for item in test_data]
    y_test = [item[1] for item in test_data]

    # Normalize training data
    X_train_normalized, means, stds = normalize(X_train)

    # Train model
    weights, bias = train_logistic_regression(
        X_train_normalized,
        y_train
    )

    # Evaluate
    test_accuracy = accuracy(
        X_test,
        y_test,
        weights,
        bias,
        means,
        stds
    )

    print(f"\nTraining samples: {len(X_train)}")
    print(f"Testing samples:  {len(X_test)}")
    print(f"Model accuracy:   {test_accuracy * 100:.2f}%")

    # Sample student
    sample_student = [8.20, 88.0, 8]

    probability = predict_probability(
        sample_student,
        weights,
        bias,
        means,
        stds
    )

    prediction = "YES" if probability >= 0.5 else "NO"

    print("\n" + "-" * 55)
    print("SAMPLE STUDENT")
    print("-" * 55)

    print(f"CGPA:        {sample_student[0]}")
    print(f"Attendance:  {sample_student[1]}%")
    print(f"CRT Score:   {sample_student[2]}/10")

    print(f"\nPredicted placement: {prediction}")
    print(f"Placement probability: {probability * 100:.2f}%")

    # Feature influence
    feature_names = [
        "CGPA",
        "Attendance",
        "CRT Score"
    ]

    print("\n" + "-" * 55)
    print("FEATURE INFLUENCE")
    print("-" * 55)

    influences = list(zip(feature_names, weights))
    influences.sort(key=lambda item: abs(item[1]), reverse=True)

    for name, influence in influences:
        direction = "positive" if influence >= 0 else "negative"
        print(f"{name}: {influence:.4f} ({direction})")

    print("\n" + "=" * 55)
    print("MODEL TRAINING COMPLETE")
    print("=" * 55)


if __name__ == "__main__":
    main()