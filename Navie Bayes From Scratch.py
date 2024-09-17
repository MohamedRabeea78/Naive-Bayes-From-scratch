import numpy as np
import pandas as pd

df = pd.DataFrame({
    'person': ['male', 'male', 'male', 'male', 'female', 'female', 'female', 'female'],
    'Height': [6, 5.92, 5.58, 5.92, 5, 5.5, 5.42, 5.75],
    'Weight': [180, 190, 170, 165, 100, 150, 130, 150],
    'FootSize': [12, 11, 12, 10, 6, 8, 7, 9]
})


def summarize_by_class(df):
    summaries = {}
    for class_value in np.unique(df['person']):
        class_data = df[df['person'] == class_value]
        summaries[class_value] = [(np.mean(class_data[column]), np.var(class_data[column]))
                                  for column in df.columns[1:]]
    return summaries

def gaussian_probability(x, mean, var):
    exponent = np.exp(-((x - mean) ** 2 / (2 * var)))
    return (1 / np.sqrt(2 * np.pi * var)) * exponent


def calculate_prior(df, class_value):
    total_count = len(df)
    class_count = len(df[df['person'] == class_value])
    return class_count / total_count

def predict(summaries, priors, input_data):
    probabilities = {}
    for class_value, class_summaries in summaries.items():
        probabilities[class_value] = priors[class_value]  
        for i in range(len(class_summaries)):
            mean, var = class_summaries[i]
            x = input_data[i]
            probabilities[class_value] *= gaussian_probability(x, mean, var)  
    return max(probabilities, key=probabilities.get)  


summaries = summarize_by_class(df)
priors = {class_value: calculate_prior(df, class_value) for class_value in np.unique(df['person'])}


try:
    height_input = float(input("Enter Height (in feet, e.g., 5.75): "))
    weight_input = float(input("Enter Weight (in pounds, e.g., 150): "))
    footsize_input = float(input("Enter Foot Size (in inches, e.g., 9): "))
except ValueError:
    print("Please enter valid numerical values.")
    exit()

input_data = [height_input, weight_input, footsize_input]

predicted_probabilities = predict(summaries, priors, input_data)


for class_value, probability in predicted_probabilities.items():
    print(f"Probability of {class_value}: {probability:.4f}")


predicted_class = max(predicted_probabilities, key=predicted_probabilities.get)
print(f"The predicted class is: {predicted_class}")


