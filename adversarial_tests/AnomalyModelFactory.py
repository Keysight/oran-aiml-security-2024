import os
import numpy as np
from sklearn.ensemble import IsolationForest, RandomForestClassifier, StackingClassifier
from sklearn.svm import SVC
from sklearn.model_selection import RandomizedSearchCV, StratifiedKFold
from sklearn.metrics import f1_score
import joblib
import logging

logger = logging.getLogger(__name__)

class AnomalyModelFactory:
    def __init__(self, model_recipe = None):
        self.model_recipe = model_recipe
        self.model = None

    @staticmethod
    def save_model(model, path="model.pkl"):
        joblib.dump(model, path)

    @staticmethod
    def load_model(path="model.pkl"):
        if not os.path.exists(path):
            raise FileNotFoundError(f"No model found at {path}. Please train the model first.")
        model = joblib.load(path)
        return model

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

    def _get_svm(self, training_data, true_anomalies):
        parameter = {
            'C': [0.1, 1, 10, 100],
            'kernel': ['linear', 'rbf'],
            'gamma': ['scale', 'auto']}
        
        svm = SVC(class_weight='balanced')
        kf = StratifiedKFold(n_splits=5, shuffle=False)
        model = RandomizedSearchCV(svm, parameter, scoring='f1', cv = kf) # verify scoring function
        md = model.fit(training_data.values, true_anomalies.values)

        return md.best_estimator_

    def build_model(self, training_data, true_anomalies, selected_model):

        logger.debug(f"selected_model: {selected_model}")

        model_dispatch = {
            "random_forest": self._get_random_forest,
            "isolation_forest": self._get_iso_forest,
            "svm": self._get_svm,
        }

        if selected_model not in model_dispatch:
            raise ValueError(f"Unknown model: {selected_model}")
        
        self.model = model_dispatch[selected_model](training_data,true_anomalies)
        return self.model
        