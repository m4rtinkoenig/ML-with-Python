import matplotlib.pyplot as plt


def plot_feature_importances(model, features):
    plt.figure(figsize=(8, 6))
    n_features = len(features)
    plt.barh(range(n_features), model.feature_importances_, align='center')
    plt.yticks(list(range(n_features)), features)
    plt.xlabel("Feature importance")
    plt.ylabel("Feature")
    plt.ylim(-1, n_features)
