import matplotlib.pyplot as plt
import xgboost as xgb


def visualize_tree_splits(model):
    xgb.plot_tree(model, num_trees=0)
    plt.rcParams['figure.figsize'] = [50, 10]
    plt.show()
