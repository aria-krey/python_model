# Модуль для предобработки данных

from sklearn.preprocessing import OneHotEncoder, RobustScaler
from sklearn.compose import ColumnTransformer
from sklearn.pipeline import Pipeline
import numpy as np

# Класс для предобработки данных
class DataPreprocessor:

    def __init__(self, categorical_features, numerical_features):
        # сохранение списков признаков
        self.categorical_features = categorical_features
        self.numerical_features = numerical_features

        # 1. Создание препроцессора - главного инструмента преобразования
        self.preprocessor = ColumnTransformer(
            transformers=[
                ('num', RobustScaler(), numerical_features), # RobustScaler вычисляет медиану и IQR для числовых признаков
                ('cat', OneHotEncoder(handle_unknown='ignore', sparse_output=False), # OneHotEncoder определяет все категории для категориальных признаков
                 categorical_features)
            ]
        )

        # 2. Инициализация переменных для кодирования меток
        self.classes_ = None
        self.class_to_index_ = None

    # Обучение препроцессора
    def fit(self, X, y=None):
        self.preprocessor.fit(X)

        if y is not None:
            # Сохраняем уникальные классы из обучающих данных
            self.classes_ = sorted(y.unique())
            self.class_to_index_ = {cls: idx for idx, cls in enumerate(self.classes_)}

        return self

    # Обучение и преобразование данных
    def fit_transform(self, X, y=None):
        # 1. Обучение и преобразование признаков
        X_transformed = self.preprocessor.fit_transform(X)

        # 2. Обработка меток классов из y и их сортировка 
        if y is not None:
            # Определяем и сохраняем классы из обучающих данных
            self.classes_ = sorted(y.unique())
            self.class_to_index_ = {cls: idx for idx, cls in enumerate(self.classes_)}

            # Кодируем целевую переменную
            y_encoded = self._encode_labels(y)

        return X_transformed, y_encoded
    
    # Преобразование данных
    def transform(self, X, y=None):
        X_transformed = self.preprocessor.transform(X)

        if y is not None:
            # Кодируем целевую переменную, заменяя неизвестные метки на -1
            y_encoded = self._encode_labels(y, handle_unknown=True)
            return X_transformed, y_encoded

        return X_transformed

    # Кодирует метки в числовой формат
    def _encode_labels(self, y, handle_unknown=False):
        if self.class_to_index_ is None: # проверка на ошибку
            raise ValueError("Сначала нужно обучить препроцессор с помощью fit или fit_transform")

        encoded = []
        unknown_count = 0

        for label in y:
            if label in self.class_to_index_:
                encoded.append(self.class_to_index_[label])
            else:
                # Если встретилась неизвестная метка
                if handle_unknown:
                    # Заменяем на -1 (можно изменить на другое значение)
                    encoded.append(-1)
                    unknown_count += 1
                else:
                    raise ValueError(f"Обнаружена неизвестная метка: {label}. " # проверка на ошибку
                                   f"Известные метки: {list(self.class_to_index_.keys())}")

        if unknown_count > 0:
            print(f" Внимание: найдено {unknown_count} записей с неизвестными метками")

        return np.array(encoded)

    # Декодирует числовые метки обратно в текстовые
    def inverse_transform(self, y_encoded):
        if self.classes_ is None: # проверка на ошибку
            raise ValueError("Препроцессор не обучен")

        # Создаем обратное отображение
        index_to_class = {idx: cls for cls, idx in self.class_to_index_.items()}

        decoded = []
        for idx in y_encoded:
            if idx == -1:
                decoded.append('unknown')
            elif idx in index_to_class:
                decoded.append(index_to_class[idx])
            else:
                decoded.append(f'unknown_{idx}')

        return np.array(decoded)

    # Получить имена признаков после преобразования
    def get_feature_names(self):
        # Для числовых признаков
        num_features = self.numerical_features

        # Для категориальных признаков после OneHot
        if hasattr(self.preprocessor.named_transformers_['cat'], 'get_feature_names_out'):
            cat_features = self.preprocessor.named_transformers_['cat'].get_feature_names_out(
                self.categorical_features
            )
        else:
            # Для старых версий sklearn
            cat_features = []
            for i, col in enumerate(self.categorical_features):
                categories = self.preprocessor.named_transformers_['cat'].categories_[i]
                for cat in categories:
                    cat_features.append(f"{col}_{cat}")

        return list(num_features) + list(cat_features)

    # Получить имена классов
    def get_class_names(self):
        if self.classes_ is None: # проверка на ошибку
            raise ValueError("Препроцессор не обучен")
        return self.classes_

    # Получить список известных классов
    def get_known_classes(self):
        if self.class_to_index_ is None: # проверка на ошибку
            raise ValueError("Препроцессор не обучен")
        return list(self.class_to_index_.keys())