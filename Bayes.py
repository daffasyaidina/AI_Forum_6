from collections import defaultdict

with open('dataset.txt') as file:
    data = [line.strip().split(',') for line in file]

features = [tuple(row[:-1]) for row in data]
outcomes = [row[-1] for row in data]

feature_counts = defaultdict(lambda: defaultdict(int))
outcome_counts = defaultdict(int)

for feature, outcome in zip(features, outcomes):
    outcome_counts[outcome] += 1
    for i, value in enumerate(feature):
        feature_counts[i, value, outcome] += 1

def naive_bayes(features):
    probabilities = {}
    total_outcomes = len(outcomes)
    for outcome in set(outcomes):
        prob_outcome = outcome_counts[outcome] / total_outcomes
        prob_features_given_outcome = 1
        for i, value in enumerate(features):
            prob_features_given_outcome *= feature_counts[i, value, outcome] / outcome_counts[outcome]
        probabilities[outcome] = prob_features_given_outcome * prob_outcome
    total_prob = sum(probabilities.values())
    for outcome in probabilities:
        probabilities[outcome] /= total_prob
    return probabilities

features_test = ('Weekday', 'No', 'Yes')
probabilities = naive_bayes(features_test)
print(f'Probability of Purchase: {probabilities["Purchase"]:.2f}')
print(f'Probability of Not purchase: {probabilities["Not purchase"]:.2f}')
