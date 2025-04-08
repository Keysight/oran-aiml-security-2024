import pandas as pd
import numpy as np
from sklearn.ensemble import IsolationForest
from sklearn.metrics import classification_report, f1_score, confusion_matrix
from art.attacks.evasion import HopSkipJump
from art.estimators.classification import BlackBoxClassifier
from a2pm import A2PMethod
import logging 

logger = logging.getLogger(__name__)

class AnomalyRobustnessEvaluator:
    def __init__(self, model, training_data, true_anomalies):
        self.model = model
        self.training_data = training_data
        self.true_anomalies = true_anomalies
        
    @staticmethod
    def perform_a2pm_attack(pattern, training_data, model):
        a2pm_method = A2PMethod(pattern)
        a2pm_method.fit(training_data.values)
        
        raw_adv_training_data = a2pm_method.generate(model, training_data.values)

        return pd.DataFrame(raw_adv_training_data, columns=training_data.columns)
    
    @staticmethod
    def predict_wrapper(model, data):
        pred = model.predict(data.values)

        if isinstance(model, IsolationForest):
            return [1 if p == -1 else 0 for p in pred]
        else: 
            return pred
    
    @staticmethod
    def get_hsja_predict(model):
        def hsja_predict(data):
            pred = model.predict(data)
            if isinstance(model, IsolationForest):
                pred = [1 if p == -1 else 0 for p in pred]
            return np.eye(2)[pred] # The output needs to be one-hot
        return hsja_predict
    
    @staticmethod
    def log_evaluation(pred, true_labels, description=""):
        inliers = sum(p == 0 for p in pred)
        outliers = sum(p == 1 for p in pred)
        cm = confusion_matrix(true_labels, pred)
        
        logger.info(f"\n--- {description} ---")
        logger.info(f"Inliers: {inliers}, Outliers: {outliers}")
        logger.info("Classification Report:\n" + classification_report(true_labels, pred, zero_division=0.0))
        logger.info(f"Macro F1: {f1_score(true_labels, pred, average='macro', zero_division=0.0):.4f}")
        logger.info(f"Confusion Matrix:\n{cm}")

    @staticmethod
    def _save_generated_dataset(dataset, path):
        if isinstance(dataset, pd.DataFrame) and isinstance(path, str):
            dataset.to_csv(path, index=False)
        else:
            raise TypeError('This method require pandas.DataFrame and string respectively')

    def test_a2pm(self, save_adv_data = False):
        if self.model is None:
            raise ValueError("Model must be trained before testing attack")

        pred = self.predict_wrapper(self.model, self.training_data)

        # TODO: Figure out patterns - how to optimize them
        # pattern = (

        #         {
        #             "type": "interval",
        #             "features": None,
        #             "ratio": 0.1,
        #             "probability": 0.6,
        #             "momentum": 0.99
        #         },
        #     )

        pattern = (

                # First pattern to be applied: Interval
                {
                    "type": "interval",
                    "features": None,
                    "integer_features": None,
                    "ratio": 0.1,
                    "max_ratio": 0.3,
                    "missing_value": 0.0,
                    "probability": 0.6,
                },

                # # Second pattern to be applied: Combination
                # {
                #     "type": "combination",
                #     "features": None,
                #     "locked_features": None,
                #     "probability": 0.4,
                # },
            )
            
        adv_training_data = self.perform_a2pm_attack(pattern, self.training_data, self.model)
        adv_pred = self.predict_wrapper(self.model, adv_training_data)

        #log results 
        self.log_evaluation(pred, self.true_anomalies, "Before A2PM Attack")
        self.log_evaluation(adv_pred, self.true_anomalies, "After A2PM Attack")

        if save_adv_data == True:
            self._save_generated_dataset(adv_training_data, './ue_a2pm')

    def test_hsja(self, save_adv_data = False):
        
        if self.model is None:
            raise ValueError("Model must be trained before testing attack")

        pred = self.predict_wrapper(self.model, self.training_data)

        clip_values = (self.training_data.min().min(), self.training_data.max().max()) # Extract minimum and maximum values
        input_shape = (self.training_data.shape[1],)
        hsja_predict = self.get_hsja_predict(self.model)
        classifier = BlackBoxClassifier(predict_fn=hsja_predict,input_shape=input_shape,nb_classes=2,clip_values=clip_values)
        hsja = HopSkipJump(classifier=classifier)

        np_adv_data = hsja.generate(self.training_data.values[:20], max_iter=50, max_eval=10000, init_eval=100, verbose=False)
        adv_data = pd.DataFrame(np_adv_data, columns=self.training_data.columns)

        adv_pred = self.predict_wrapper(self.model, adv_data)

        #log results 
        self.log_evaluation(pred, self.true_anomalies, "Before HSJA Attack")
        self.log_evaluation(adv_pred, self.true_anomalies[:len(adv_pred)], "After HSJA Attack")

        logger.info(adv_data.compare(self.training_data[:len(adv_pred)]))
        logger.info(f"L1 Norm {np.linalg.norm(adv_data - self.training_data[:len(adv_pred)], ord=1, axis=0)}")
        logger.info(f"L2 Norm { np.linalg.norm(adv_data - self.training_data[:len(adv_pred)], ord=2, axis=0)}")
        logger.info(f"L_inf Norm {np.linalg.norm(adv_data - self.training_data[:len(adv_pred)], ord=np.inf, axis=0)}")

        if save_adv_data == True:
            self._save_generated_dataset(adv_data, './ue_hsja')


