import numpy as np
import matplotlib.pyplot as plt

from sklearn import tree
from sklearn.metrics import accuracy_score
from sklearn.model_selection import StratifiedKFold


def evaluatePerformance():
    '''
    Evaluate the performance of decision trees,
    averaged over 100 trials of 10-fold cross-validation
    
    Return:
      a matrix giving the performance that will contain the following entries:
      stats[0,0] = mean accuracy of decision tree
      stats[0,1] = std deviation of decision tree accuracy
      stats[1,0] = mean accuracy of decision stump
      stats[1,1] = std deviation of decision stump accuracy
      stats[2,0] = mean accuracy of 3-level decision tree
      stats[2,1] = std deviation of 3-level decision tree
      
    ** Note that your implementation must follow this API**
    '''
    
    # Load Data
    filename = './data/SPECTF.dat'
    data = np.loadtxt(filename, delimiter=',')
    X = data[:, 1:]
    y = np.array([data[:, 0]]).T
    n, d = X.shape

    # create an array to store accuracy scores for each trial
    decisionTreeScores = np.zeros(1000)
    decisionStumpScores = np.zeros(1000)
    dt3Scores = np.zeros(1000)

    for trial in range(100):
        # shuffle the data at the start of each trial
        idx = np.arange(n)
        np.random.seed(trial)
        np.random.shuffle(idx)
        X = X[idx]
        y = y[idx]

        # perform 10-fold cross-validation for each trial
        skf = StratifiedKFold(n_splits=10, shuffle=False)

        fold = 0
        for train_index, test_index in skf.split(X, y):
            X_train, X_test = X[train_index], X[test_index]
            y_train, y_test = y[train_index], y[test_index]

            # train the decision tree
            decisionTree = tree.DecisionTreeClassifier()
            decisionTree.fit(X_train, y_train)

            # evaluate the trained model on the test fold
            y_pred = decisionTree.predict(X_test)
            decisionTreeScores[trial * 10 + fold] = accuracy_score(y_test, y_pred)

            # train the decision stump
            decisionStump = tree.DecisionTreeClassifier(max_depth=1)
            decisionStump.fit(X_train, y_train)

            # evaluate the trained model on the test fold
            y_pred = decisionStump.predict(X_test)
            decisionStumpScores[trial * 10 + fold] = accuracy_score(y_test, y_pred)

            # train the 3-level decision tree
            dt3 = tree.DecisionTreeClassifier(max_depth=3)
            dt3.fit(X_train, y_train)

            # evaluate the trained model on the test fold
            y_pred = dt3.predict(X_test)
            dt3Scores[trial * 10 + fold] = accuracy_score(y_test, y_pred)

            fold += 1

    # compute mean and standard deviation of accuracy scores
    meanDecisionTreeAccuracy = np.mean(decisionTreeScores)
    stddevDecisionTreeAccuracy = np.std(decisionTreeScores)

    meanDecisionStumpAccuracy = np.mean(decisionStumpScores)
    stddevDecisionStumpAccuracy = np.std(decisionStumpScores)

    meanDT3Accuracy = np.mean(dt3Scores)
    stddevDT3Accuracy = np.std(dt3Scores)

    # make certain that the return value matches the API specification
    stats = np.zeros((3, 2))
    stats[0, 0] = meanDecisionTreeAccuracy
    stats[0, 1] = stddevDecisionTreeAccuracy
    stats[1, 0] = meanDecisionStumpAccuracy
    stats[1, 1] = stddevDecisionStumpAccuracy
    stats[2, 0] = meanDT3Accuracy
    stats[2, 1] = stddevDT3Accuracy

    return stats, decisionTreeScores, decisionStumpScores, dt3Scores


# Do not modify from HERE...
if __name__ == "__main__":
    stats, decisionTreeScores, decisionStumpScores, dt3Scores = evaluatePerformance()

    # Plot the learning curve
    training_sizes = np.linspace(0.1, 1.0, 10)
    mean_accuracies = np.zeros((3, 10))
    std_deviations = np.zeros((3, 10))

    for i, size in enumerate(training_sizes):
        indices = np.random.choice(len(decisionTreeScores), int(size * len(decisionTreeScores)), replace=False)

        mean_accuracies[0, i] = np.mean(decisionTreeScores[indices])
        std_deviations[0, i] = np.std(decisionTreeScores[indices])

        mean_accuracies[1, i] = np.mean(decisionStumpScores[indices])
        std_deviations[1, i] = np.std(decisionStumpScores[indices])

        mean_accuracies[2, i] = np.mean(dt3Scores[indices])
        std_deviations[2, i] = np.std(dt3Scores[indices])

    plt.errorbar(training_sizes, mean_accuracies[0], yerr=std_deviations[0], label='Decision Tree')
    plt.errorbar(training_sizes, mean_accuracies[1], yerr=std_deviations[1], label='Decision Stump')
    plt.errorbar(training_sizes, mean_accuracies[2], yerr=std_deviations[2], label='3-level Decision Tree')

    plt.xlabel('Training Data Size')
    plt.ylabel('Accuracy')
    plt.title('Learning Curve')
    plt.legend()
    plt.show()

    print("Decision Tree Accuracy = ", stats[0, 0], " (", stats[0, 1], ")")
    print("Decision Stump Accuracy = ", stats[1, 0], " (", stats[1, 1], ")")
    print("3-level Decision Tree = ", stats[2, 0], " (", stats[2, 1], ")")
# ...to HERE.