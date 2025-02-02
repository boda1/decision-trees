import pandas as pd
import numpy as np
from decision_tree.decision_tree import DecisionTree

class RandomForestClassifier():
    """
    Docstring
    """

    def __init__(self, 
                 max_depth: int,
                 min_information_gain: float,
                 min_samples_per_leaf: int,
                 feature_selection: str,
                 n_learners: int,
                 bootstrap_sample_size: int,
                 bootstrap: bool,
                 oob_score: bool
                ):
        self.max_depth = max_depth
        self.min_information_gain = min_information_gain
        self.min_samples_per_leaf = min_samples_per_leaf
        self.feature_selection = feature_selection
        self.n_learners = n_learners
        self.bootstrap_sample_size = bootstrap_sample_size
        self.bootstrap = bootstrap
        self.oob_score = oob_score
        self.trained_base_learners = []

    def create_bootstrap_data(self, X: pd.DataFrame, Y: pd.DataFrame) -> list: # ToDo: Check whether df or series
        bootstrap_samples_X = []
        bootstrap_samples_Y = []
        
        for idx in range(self.n_learners):
            sample_row_idx = np.random.choice(X.shape[0], size=self.bootstrap_sample_size, replace=True)
            bootstrap_samples_X.append(X.iloc[sample_row_idx, :])
            bootstrap_samples_Y.append(Y.iloc[sample_row_idx, :])

        return bootstrap_samples_X, bootstrap_samples_Y
    
    def train(self, train_X: pd.DataFrame, train_Y: pd.DataFrame) -> None:
        """
        Training a decision tree for every bootstrapped sample and storing in trained_base_learners attribute
        """
    
        bootstrap_samples_X, bootstrap_samples_Y = self.create_bootstrap_data(train_X, train_Y)
        
        for idx, _ in enumerate(bootstrap_samples_X):
            base_learner = DecisionTree(self.max_depth, self.min_information_gain, self.min_samples_per_leaf, self.feature_selection)
            base_learner.train(bootstrap_samples_X[idx], bootstrap_samples_Y[idx])
            self.trained_base_learners.append(base_learner)
    
    def predict_one_sample(self, row_data: pd.Series):
        predictions = []
        
        for base_learner in self.trained_base_learners:
            predicted_probabilities = base_learner.predict_prob_one_sample(row_data)
            predicted_classes = max(predicted_probabilities, key=predicted_probabilities.get)
            predictions.append(predicted_classes)

        return max(set(predictions), key=predictions.count)
        
    def predict(self, data: pd.DataFrame) -> pd.DataFrame:
        return data.apply(self.predict_one_sample, axis=1)
