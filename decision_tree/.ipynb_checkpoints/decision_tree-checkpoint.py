import pandas as pd
import numpy as np
from collections import defaultdict
from decision_tree.tree_node import TreeNode

"""
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
"""

class DecisionTree():
    """
    Decision tree classifier
    Train using 'train' method 
    Predict using 'predict' method
    Using Gini impurity for quantifying node purity
    """ 
    
    def __init__(self,
                 max_depth: int,
                 min_information_gain: float,
                 min_samples_per_leaf: int,
                 feature_selection: str = None,
                ):
        self.max_depth = max_depth
        self.min_information_gain = min_information_gain
        self.min_samples_per_leaf = min_samples_per_leaf
        self.feature_selection = feature_selection
        self.tree = None

    def get_labels(self, Y_data: pd.DataFrame) -> list:
        """
        return unique labels from Y (target) data
        """
        if Y_data.shape[0] > 1:
            return list(Y_data.iloc[:, -1].unique())    
        else:
            return list(Y_data.iloc[:, 0].unique())

    def select_features(self, data: pd.DataFrame) -> list:
        """
        Perform feature selection using technique from hyperparameter "feature_selection"
        """
        feature_idx = list(range(data.shape[1] - 1))

        if self.feature_selection == "sqrt":
            return np.random.choice(feature_idx, size=int(np.sqrt(len(feature_idx))))
        elif self.feature_selection == "log":
            return np.random.choice(feature_idx, size=int(np.log2(len(feature_idx))))
        else:
            return feature_idx  

    def get_label_probas(self, data: pd.DataFrame):
        """
        calculate likelihood of given label appearing in dataset
        """
        labels = self.get_labels(data)
        label_probas = defaultdict(float)
        for label in labels:
            label_mask = data.iloc[:, -1] == label
            label_probas[label] =  data[label_mask].shape[0] / data.shape[0]
        
        return label_probas

    def gini_impurity(self, split_groups: list) -> float:
        """
        return Gini impurity of a given data set or groups of data resulting from a split
        """
        classes = set([item for group in split_groups for item in self.get_labels(group)])
        total_size = sum([group.shape[0] for group in split_groups])
        gini_score = float(0.0)
        for group in split_groups:
            score = 1.0
            group_size = group.shape[0]
            if group_size == 0:
                continue
            for class_val in classes:
                mask = group.iloc[:, -1] == class_val
                class_size = group[mask].shape[0]
                proportion = class_size / group_size
                score -= proportion ** 2
            gini_score += score * (group_size / total_size)
        return gini_score
    
    def split_data(self, data: pd.DataFrame, feature_idx: int, split_threshold: float):    
        below_threshold = data.iloc[:, feature_idx] <= split_threshold
        split_above = data[~below_threshold]
        split_below = data[below_threshold]
    
        return split_above, split_below
    
    def best_split(self, data: pd.DataFrame) -> pd.DataFrame:
        min_impurity_val = 1e6
        min_impurity_feature_idx = 0
        min_impurity_feature_value = 0
        classes = self.get_labels(data) 
        feature_idx = self.select_features(data)
        
        for idx in feature_idx:
            split_thresholds = np.percentile(data.iloc[:, idx], q=np.arange(25, 100, 25))
            for split_threshold in split_thresholds:
                split_left, split_right = self.split_data(data, idx, split_threshold)
                impurity = self.gini_impurity([split_left, split_right])
                if impurity < min_impurity_val:
                    min_impurity_val = impurity
                    min_impurity_feature_idx = idx
                    min_impurity_split_threshold = split_threshold
                    split_left_min_impurity, split_right_min_impurity = split_left, split_right
        
        return split_left_min_impurity, split_right_min_impurity, min_impurity_feature_idx, min_impurity_split_threshold, min_impurity_val

    def create_tree(self, data: pd.DataFrame, current_depth=0):
        if current_depth >= self.max_depth:
            return None
    
        # find best split
    
        split_left_data, split_right_data, min_impurity_feature_idx, min_impurity_split_threshold, split_impurity_val = self.best_split(data)
        
        # find information gain from splitting
    
        node_impurity = self.gini_impurity([data])
        information_gain = node_impurity - split_impurity_val
    
        # get label probabilities for current node
    
        label_probabilities = self.get_label_probas(data)
        
        # create internal node
    
        node = TreeNode(data=data, feature_idx=min_impurity_feature_idx, split_threshold=min_impurity_split_threshold, label_probabilities=label_probabilities, information_gain=information_gain)
    
        # check whether conditions for stopping are met
    
        if split_left_data.shape[0] < self.min_samples_per_leaf or split_right_data.shape[0] < self.min_samples_per_leaf:
            return node
        elif information_gain <= self.min_information_gain:
            return node
        
        # if stopping conditions are not met continue recursing through tree
    
        current_depth += 1
        node.left = self.create_tree(split_left_data, current_depth)
        node.right = self.create_tree(split_right_data, current_depth)

        return node

    def predict_prob_one_sample(self, data: pd.Series):
        node = self.tree
        
        while node:
            prediction = node.label_probabilities
            idx = node.feature_idx 
            if data.iloc[idx] < node.split_threshold:
                node = node.left
            else:
                node = node.right
    
        return prediction
    
    def predict_prob_all_samples(self, data: pd.DataFrame):
        """
        Returns a dict with classes and probabilities as keys and labels respectivel for each row
        """
        prediction_probabilities = data.apply(self.predict_prob_one_sample, axis=1)
    
        return prediction_probabilities

    def predict(self, X_data: pd.DataFrame) -> pd.DataFrame:
        """
        Returns the class with maximum probability for each row
        """
        probabilities = self.predict_prob_all_samples(X_data)
        predictions = pd.DataFrame(probabilities.apply(lambda x: max(x, key=x.get)))
        
        return predictions

    def train(self, X_train: pd.DataFrame, Y_train: pd.DataFrame) -> None:
        """
        Train model using variables (X data) and targets (Y data)
        """
        # combine features and labels
        train_labels = self.get_labels(Y_train)
        train_data = pd.concat([X_train, Y_train], axis=1)
    
        # begin creating the tree
        self.tree = self.create_tree(data=train_data, current_depth=0)