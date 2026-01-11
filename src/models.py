# Модуль с моделями машинного обучения

import pandas as pd
import numpy as np

from sklearn.linear_model import LogisticRegression
from sklearn.ensemble import RandomForestClassifier, GradientBoostingClassifier
from sklearn.multiclass import OneVsRestClassifier
from sklearn.neighbors import KNeighborsClassifier
from sklearn.tree import DecisionTreeClassifier
from sklearn.model_selection import cross_val_score, StratifiedKFold
from sklearn.metrics import accuracy_score, confusion_matrix, classification_report
from sklearn.metrics import f1_score

class ModelTrainer:
    """
    Класс для обучения, оценки и сравнения моделей
    """

    def __init__(self, random_state=42):
        self.random_state = random_state
        self.models = {}
        self.results = {}

    # --------------------------------------------------
    # Создание моделей
    # --------------------------------------------------
    def get_models(self):
        return {
            'Decision Tree': DecisionTreeClassifier(random_state=self.random_state),
            'Random Forest': RandomForestClassifier(random_state=self.random_state),
            'Gradient Boosting': GradientBoostingClassifier(random_state=self.random_state),
            'KNN': KNeighborsClassifier()
        }

    # --------------------------------------------------
    # Обучение и оценка моделей
    # --------------------------------------------------
    def train_models(self, X_train, y_train, X_test, y_test, model_params=None):
        models = self.get_models()

        # Применяем пользовательские параметры
        if model_params:
            for name, params in model_params.items():
                if name in models:
                    models[name].set_params(**params)

        for name, model in models.items():
            print(f"\n{'=' * 50}")
            print(f"Обучение модели: {name}")
            print(f"{'=' * 50}")

            # Обучение
            model.fit(X_train, y_train)
            self.models[name] = model

            # Предсказания
            y_pred = model.predict(X_test)
            test_f1 = f1_score(y_test, y_pred, average='macro')

            # Кросс-валидация
            cv = StratifiedKFold(n_splits=3, shuffle=True, random_state=self.random_state)
            cv_scores = cross_val_score(
                model,
                X_train,
                y_train,
                cv=cv,
                scoring='f1_macro'
            )

            cv_mean = cv_scores.mean()
            test_accuracy = accuracy_score(y_test, y_pred)

            # Сохранение результатов
            self.results[name] = {
                'model': model,
                'test_accuracy': test_accuracy,
                'test_f1': test_f1,          # ← ВАЖНО
                'cv_mean': cv_mean,
                'predictions': y_pred,
                'report': classification_report(y_test, y_pred, output_dict=True)
            }


            print(f"CV F1 (train): {cv_mean:.4f}")
            print(f"Точность на тесте: {test_accuracy:.4f}")
            print(f"F1 macro на тесте: {test_f1:.4f}")
            print(f"Разница: {abs(cv_mean - test_accuracy):.4f}")

            if abs(cv_mean - test_accuracy) > 0.05:
                print("⚠ Возможное переобучение")
            else:
                print("✓ Модель устойчива")

        return self.results

    # --------------------------------------------------
    # Лучшая модель
    # --------------------------------------------------
    def get_best_model(self):
        if not self.results:
            raise ValueError("Сначала обучите модели!")

        best_model_name = max(
            self.results.items(),
            key=lambda x: x[1]['test_f1']
        )[0]

        return best_model_name, self.results[best_model_name]

    # --------------------------------------------------
    # Сравнение моделей
    # --------------------------------------------------
    def compare_models(self):
        comparison_data = []

        for name, result in self.results.items():
            cv_value = result.get('cv_mean')

            comparison_data.append({
                'Model': name,
                'Train CV (F1)': round(result['cv_mean'], 4),
                'Test F1': round(result['test_f1'], 4),
                'Difference': round(abs(result['cv_mean'] - result['test_f1']), 4)
            })


        return pd.DataFrame(comparison_data).sort_values(
            'Test F1', ascending=False
        )


    # --------------------------------------------------
    # Детальный отчет
    # --------------------------------------------------
    def get_detailed_report(self, y_test, y_pred):
        report = classification_report(y_test, y_pred, output_dict=True)
        df_report = pd.DataFrame(report).transpose()
        cm = confusion_matrix(y_test, y_pred)
        return df_report, cm
