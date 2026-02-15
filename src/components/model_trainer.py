import os
import sys
from dataclasses import dataclass

from catboost import CatBoostRegressor

from sklearn.ensemble import (
    AdaBoostRegressor,
    GradientBoostingRegressor,
    RandomForestRegressor,
)

from sklearn.linear_model import LinearRegression
from sklearn.metrics import r2_score
from sklearn.neighbors import KNeighborsRegressor
from sklearn.tree import DecisionTreeRegressor
from xgboost import XGBRegressor

from src.exception import CustomException
from src.logger import logging

from src.utils import save_object, evaluate_models


@dataclass
class ModelTrainerConfig:
    trained_model_file_path = os.path.join("artifacts", "model.pkl")


class ModelTrainer:
    def __init__(self):
        self.model_trainer_config = ModelTrainerConfig()

    def initiate_model_trainer(self, train_array, test_array):
        try:
            logging.info("Spliting training and test input data ")

            X_train, y_train, X_test, y_test = (
                train_array[:, :-1],
                train_array[:, -1],
                test_array[:, :-1],
                test_array[:, -1],
            )

            models = {
                "Random Forest": RandomForestRegressor(),
                "Decision Tree": DecisionTreeRegressor(),
                "Gradient Boosting": GradientBoostingRegressor(),
                "Linear Regression": LinearRegression(),
                "K-Neighbors": KNeighborsRegressor(),
                "XGB": XGBRegressor(),
                "CatBoost": CatBoostRegressor(verbose=False),
                "AdaBoost": AdaBoostRegressor(),
            }
            params = {
                "Random Forest": {"n_estimators": [8, 16, 32, 64, 128]},
                "Decision Tree": {
                    "criterion": [
                        "squared_error",
                        "friedman_mse",
                        "absolute_error",
                        "poisson",
                    ]
                },
                "Gradient Boosting": {
                    "learning_rate": [0.1, 0.01, 0.05],
                    "subsample": [0.6, 0.8, 0.9],
                    "n_estimators": [16, 32, 64],
                },
                "Linear Regression": {},
                "K-Neighbors": {"n_neighbors": [3, 5, 7]},
                "XGB": {
                    "learning_rate": [0.1, 0.01],
                    "n_estimators": [16, 32, 64],
                },
                "CatBoost": {
                    "depth": [6, 8],
                    "learning_rate": [0.01, 0.1],
                    "iterations": [50, 100],
                },
                "AdaBoost": {
                    "learning_rate": [0.1, 0.01],
                    "n_estimators": [16, 32, 64],
                },
            }

            models_report: dict = evaluate_models(
                X_train=X_train,
                y_train=y_train,
                X_test=X_test,
                y_test=y_test,
                models=models,
                param=params,
            )

            best_model_score = max(models_report.values())
            best_model_name = max(models_report, key=models_report.get)

            best_model = models[best_model_name]

            if best_model_score < 0.6:
                raise CustomException("No best model found", sys)

            save_object(
                file_path=self.model_trainer_config.trained_model_file_path,
                obj=best_model,
            )

            predicted = best_model.predict(X_test)
            r2_square = r2_score(y_test, predicted)

            return r2_square

        except Exception as e:
            raise CustomException(e, sys)
