import pandas as pd

class TreeNode():
    def __init__(self, data: pd.DataFrame, feature_idx: int, split_threshold: float, label_probabilities: float, information_gain: float):
        self.data = data
        self.feature_idx = feature_idx
        self.split_threshold = split_threshold
        self.label_probabilities = label_probabilities
        self.information_gain = information_gain
        self.left = None
        self.right = None

    def node_details(self):
        if self.left or self.right:
            return f"Feature used to split: {self.data[:, self.feature_idx]}, split threshold: {self.split_threshold}, information gain: {self.information_gain}"