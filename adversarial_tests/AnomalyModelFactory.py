import os
import numpy as np
from sklearn.ensemble import IsolationForest, RandomForestClassifier
from sklearn.model_selection import RandomizedSearchCV
from sklearn.metrics import f1_score
import joblib
import logging

logger = logging.getLogger(__name__)

class AnomalyModelFactory:
    def __init__(self, model_recipe = None):
        self.model_recipe = model_recipe
        self.model = None

    def save_model(self, path="model.pkl"):
        joblib.dump(self.model, path)

    def load_model(self, path="model.pkl"):
        if not os.path.exists(path):
            raise FileNotFoundError(f"No model found at {path}. Please train the model first.")
        self.model = joblib.load(path)
        return self.model

    @staticmethod
    def get_scorer(true_labels):
        def scorer(estimator, X):
            pred = estimator.predict(X)
            pred = [1 if p == -1 else 0 for p in pred]
            return f1_score(true_labels, pred)
        return scorer
        
    def _get_iso_forest(self, training_data, true_anomalies):
        random_state = 4
        parameter = {'contamination': [of for of in np.arange(0.01, 0.5, 0.02)],
                     'n_estimators': [100*(i+1) for i in range(1, 10)],
                     'max_samples': [0.005, 0.01, 0.1, 0.15, 0.2, 0.3, 0.4]}
        cv = [(slice(None), slice(None))]
        scorer = self.get_scorer(true_anomalies)
        iso = IsolationForest(random_state=random_state, bootstrap=True, warm_start=False)
        model = RandomizedSearchCV(iso, parameter, scoring=scorer, cv=cv, n_iter=50)
        md = model.fit(training_data.values)
        return md.best_estimator_
    
    def _get_random_forest(self, training_data, true_anomalies):
        random_state = 4
        parameter = {'n_estimators': [100*(i+1) for i in range(1, 12)],
                     'max_samples': [0.005, 0.01, 0.1, 0.15, 0.2, 0.3, 0.4],
                     'criterion': ["gini" , "entropy", "log_loss"]}
        random_forest = RandomForestClassifier(random_state=random_state, bootstrap=True, warm_start=False)
        model = RandomizedSearchCV(random_forest, parameter, scoring=None, cv=None, n_iter=50)
        md = model.fit(training_data.values, true_anomalies.values)
        return md.best_estimator_
        
    def build_model(self, training_data, true_anomalies, selected_model):

        logger.debug(f"selected_model: {selected_model}")

        model_dispatch = {
            "random_forest": self._get_random_forest,
            "isolation_forest": self._get_iso_forest,
        }

        if selected_model not in model_dispatch:
            raise ValueError(f"Unknown model: {selected_model}")
        
        self.model = model_dispatch[selected_model](training_data,true_anomalies)
        return self.model
        