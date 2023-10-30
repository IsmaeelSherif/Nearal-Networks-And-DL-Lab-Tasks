import numpy as np

def confusion_matrix(actual, predicted, num_classes):
    # Initialize a num_classes x num_classes matrix with zeros
    cm = np.zeros((num_classes, num_classes))

    # Calculate confusion matrix
    for a, p in zip(actual, predicted):
        cm[a][p] += 1

    return cm

# Example of predicted and actual labels for a 3-class classification problem
predicted = np.array([0, 1, 1, 2, 0, 2, 1, 0, 2, 0])
actual = np.array([0, 1, 1, 2, 0, 1, 2, 1, 2, 0])

# Number of classes in the classification
num_classes = 3

# Calculating the confusion matrix
cm = confusion_matrix(actual, predicted, num_classes)

print("Confusion Matrix:")
print(cm)