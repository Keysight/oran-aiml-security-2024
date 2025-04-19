import pandas as pd
import numpy as np
from sklearn.ensemble import IsolationForest
from sklearn.metrics import classification_report, f1_score, roc_auc_score, precision_recall_curve, confusion_matrix
from art.attacks.evasion import HopSkipJump
from art.estimators.classification import BlackBoxClassifier
from a2pm import A2PMethod
import logging 
import time

logger = logging.getLogger(__name__)

class AnomalyRobustnessEvaluator:
    def __init__(self):
        pass
        
    @staticmethod
    def perform_a2pm_attack(pattern, data, model, y = None, y_target = None):
        a2pm_method = A2PMethod(pattern)
        
        raw_adv_training_data = a2pm_method.fit_generate(model, data.values, y = y, y_target= y_target)

        return pd.DataFrame(raw_adv_training_data, columns=data.columns, index=data.index)
    
    @staticmethod
    def predict_wrapper(model, data):
        start_time = time.time()
        pred = model.predict(data.values)
        predict_time = time.time() - start_time

        if isinstance(model, IsolationForest):
            return [1 if p == -1 else 0 for p in pred], predict_time
        else: 
            return pred, predict_time
    
    @staticmethod
    def get_hsja_predict(model):
        def hsja_predict(data):
            pred = model.predict(data)
            if isinstance(model, IsolationForest):
                pred = [1 if p == -1 else 0 for p in pred]
            return np.eye(2)[pred] # The output needs to be one-hot
        return hsja_predict
    
    @staticmethod
    def log_evaluation(pred, true_labels, predict_time):
        inliers = sum(p == 0 for p in pred)
        outliers = sum(p == 1 for p in pred)
        cm = confusion_matrix(true_labels, pred)

        logger.info(f"Inliers: {inliers}, Outliers: {outliers}")
        logger.info("Classification Report:\n" + classification_report(true_labels, pred, zero_division=0.0))
        logger.info(f"Macro F1: {f1_score(true_labels, pred, average='macro', zero_division=0.0):.4f}")
        logger.info(f"Confusion Matrix:\n{cm}")
        logger.info(f"Prediction time:  {predict_time}")
        logger.info("\n")

    @staticmethod
    def _save_generated_dataset(dataset, path):
        if isinstance(dataset, pd.DataFrame) and isinstance(path, str):
            dataset.to_csv(path, index=False)
        else:
            raise TypeError('This method require pandas.DataFrame and string respectively')
        
    @staticmethod
    def model_timing_benchmark(test_data, repeats = 100, *args):
        for model in args:
            times = []

            for _ in range(repeats):
                start = time.perf_counter()
                model.predict(test_data.values)
                end = time.perf_counter() 
                times.append(end - start) 

            avarage_time = np.mean(times)
            min = np.min(times)
            max = np.max(times)
            model_name = model.__class__.__name__
            logger.info(f"Model: {model_name} Avg time: {avarage_time} sec avaraged over {repeats} repeats with min: {min} sec and max: {max} sec")

    def run_all_tests(self, model, test_data, true_anomalies, pattern, save_adv_data = False,):
        self.test_clean(model, test_data, true_anomalies)
        self.test_a2pm(model, test_data, true_anomalies, pattern, save_adv_data)
        self.test_hsja(model, test_data, true_anomalies, save_adv_data)

    def test_clean(self, model, test_data, true_anomalies):
        if model is None:
            raise ValueError("Model must be trained before testing attack")
        
        if not isinstance(test_data, pd.DataFrame) and not isinstance(true_anomalies, pd.DataFrame):
            raise TypeError
        
        description = "Clean Attack" 
        logger.info(f"----{description}-----")
        logger.info(f"Model: {model.__class__.__name__}")

        pred, predict_time = self.predict_wrapper(model, test_data)

        self.log_evaluation(pred, true_anomalies, predict_time)

    def test_a2pm(self, model, test_data, true_anomalies, pattern, save_adv_data = False):
        if model is None:
            raise ValueError("Model must be trained before testing attack")
        
        if not isinstance(test_data, pd.DataFrame) and not isinstance(true_anomalies, pd.DataFrame):
            raise TypeError
        
        description = "A2PM Attack" 
        logger.info(f"----{description}-----")
        logger.info(f"Model: {model.__class__.__name__}")

        adv_training_data = self.perform_a2pm_attack(pattern, test_data, model)
        adv_pred, adv_predict_time = self.predict_wrapper(model, adv_training_data)

        self.log_evaluation(adv_pred, true_anomalies, adv_predict_time)

        if save_adv_data == True:
            self._save_generated_dataset(adv_training_data, './ue_a2pm')

    def test_hsja(self, model, test_data, true_anomalies, save_adv_data = False, number_of_samples = 20):
        if not isinstance(test_data, pd.DataFrame) and not isinstance(true_anomalies, pd.DataFrame):
            raise TypeError

        if model is None:
            raise ValueError("Model must be trained before testing attack")
        
        description = "HSJA Attack" 
        logger.info(f"----{description}-----")
        logger.info(f"Model: {model.__class__.__name__}")

        clip_values = (test_data.min().min(), test_data.max().max()) # Extract minimum and maximum values
        input_shape = (test_data.shape[1],)
        hsja_predict = self.get_hsja_predict(model)
        classifier = BlackBoxClassifier(predict_fn=hsja_predict,input_shape=input_shape,nb_classes=2,clip_values=clip_values)
        hsja = HopSkipJump(classifier=classifier)

        np_adv_data = hsja.generate(test_data.values[:number_of_samples], max_iter=25, max_eval=1000, init_eval=50, verbose=False)
        adv_data = pd.DataFrame(np_adv_data, columns=test_data.columns, index=test_data.index[:number_of_samples])

        adv_pred, adv_predict_time = self.predict_wrapper(model, adv_data)

        self.log_evaluation(adv_pred, true_anomalies[:len(adv_pred)], adv_predict_time) # TODO: fix timing comparison - predict_time predicts on whole dataset, while adv. predic on X samples
        logger.info(f"L1 Norm {np.linalg.norm(adv_data - test_data[:len(adv_data)], ord=1, axis=0)}")
        logger.info(f"L2 Norm { np.linalg.norm(adv_data - test_data[:len(adv_data)], ord=2, axis=0)}")
        logger.info(f"L_inf Norm {np.linalg.norm(adv_data - test_data[:len(adv_data)], ord=np.inf, axis=0)}")
    
        if save_adv_data == True:
            self._save_generated_dataset(adv_data, './ue_hsja')


