# Модуль с моделями машинного обучения

from sklearn.linear_model import LogisticRegression
from sklearn.ensemble import RandomForestClassifier, GradientBoostingClassifier
from sklearn.svm import SVC
from sklearn.multiclass import OneVsRestClassifier
from sklearn.neighbors import KNeighborsClassifier
from sklearn.tree import DecisionTreeClassifier
from sklearn.model_selection import GridSearchCV, cross_val_score, StratifiedKFold
from sklearn.metrics import (accuracy_score, precision_score, recall_score,
                             f1_score, confusion_matrix, classification_report)
import pandas as pd
import numpy as np

# Класс для обучения и оценки моделей
class ModelTrainer:

    def __init__(self, random_state=42):
        self.random_state = random_state
        self.models = {}
        self.results = {}

    # Возвращает словарь моделей для сравнения
    def get_models(self):
        models = {
            'Logistic Regression': OneVsRestClassifier( LogisticRegression(
                max_iter=200,
                random_state=self.random_state,
                C=0.5,                     
                class_weight='balanced',   
                solver='liblinear',
                tol = 1e-3,
                warm_start = False
            )
            ),
            'Random Forest': RandomForestClassifier(
                n_estimators=50,           
                max_depth=8,             
                min_samples_split=20,           
                max_features='sqrt',          
                class_weight='balanced',
                random_state=42
            ),
            'Gradient Boosting': GradientBoostingClassifier(
                n_estimators=30,          
                max_depth=2,               
                learning_rate=0.15,        
                subsample=1.0,
                random_state = 42
            ),
            'Decision Tree': DecisionTreeClassifier(
                max_depth=6,            
                min_samples_leaf=30,        
                class_weight='balanced',
                random_state = 42
            ),
            'KNN': KNeighborsClassifier(
                n_neighbors=25,           
                weights='distance',    
            )
        }
        return models

    # Обучает и оценивает все модели
    def train_models(self, X_train, y_train, X_test, y_test):
        # 1. Получение словаря моделей для сравнения
        models = self.get_models()

        # 2. Проход по каждой модели в словаре
        for name, model in models.items():
            print(f"\n{'=' * 50}")
            print(f"Обучение модели: {name}")
            print(f"{'=' * 50}")

            # 3. Обучение
            model.fit(X_train, y_train)
            self.models[name] = model

            # 4. Предсказания
            y_pred = model.predict(X_test)
            cv = StratifiedKFold(n_splits=3, shuffle=True, random_state=42)
            cv_scores = cross_val_score(
                model, X_train, y_train,
                cv=cv,
                scoring='f1_macro'
            )

            # 5. Оценка
            test_accuracy = accuracy_score(y_test, y_pred)
            cv_mean = cv_scores.mean()
    

            # 6. Сохранение результатов
            self.results[name] = {
                'model': model,
                'test_accuracy': test_accuracy,
                'cv_f1': cv_mean,
                'predictions': y_pred,
                'report': classification_report(y_test, y_pred, output_dict=True)
            }

            print(f'CV F! (train): {cv_mean:.4f}')
            print(f"Точность на тесте:    {test_accuracy:.4f}")
            print(f"Разница: {abs(cv_mean - test_accuracy):.4f}")

            # Проверка на переобучение
            if abs(cv_mean - test_accuracy) > 0.05:
                print("Внимание: возможное переобучение")
            else:
                print("Модель устойчива")

        return self.results

    # Возвращает лучшую модель по точности на тесте
    def get_best_model(self):
        if not self.results: # проверка на ошибку
            raise ValueError("Сначала обучите модели!")

        best_model_name = max(self.results.items(),
                              key=lambda x: x[1]['test_accuracy'])[0]

        return best_model_name, self.results[best_model_name]

    # Сравнивает все модели и возвращает DataFrame
    def compare_models(self):
        comparison_data = []

        for name, result in self.results.items(): # проход по всем результатам
            cv_value = result.get('cv_mean')

            comparison_data.append({
                'Model': name,
                'Train CV (F1)': round(cv_value, 4) if cv_value is not None else '—',
                'Test Accuracy': round(result['test_accuracy'], 4),
                'Difference': round(abs(cv_value - result['test_accuracy']), 4)
                if cv_value is not None else '—'
            })

        return pd.DataFrame(comparison_data).sort_values('Test Accuracy', ascending=False)

    # Возвращает детальный отчет по модели
    def get_detailed_report(self, model_name, y_test, y_pred):
        report = classification_report(y_test, y_pred, output_dict=True)
        df_report = pd.DataFrame(report).transpose()

        # Матрица ошибок
        cm = confusion_matrix(y_test, y_pred)

        return df_report, cm