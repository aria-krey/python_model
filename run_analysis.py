"""
Главный файл для запуска полного анализа NSL-KDD
"""
import sys
import os
sys.path.append(os.path.dirname(os.path.abspath(__file__)))

import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
from src.data_loader import NSLKDDDataLoader
from src.preprocessor import DataPreprocessor
from src.models import ModelTrainer
from sklearn.metrics import confusion_matrix
import warnings
warnings.filterwarnings('ignore')

# Настройка отображения
plt.rcParams['figure.figsize'] = (12, 8)
plt.style.use('seaborn-v0_8-darkgrid')
sns.set_palette("husl")

def main():
    """Основная функция для запуска всего кода"""
    print("=" * 70)
    print("КЛАССИФИКАЦИЯ СЕТЕВОГО ТРАФИКА НА ОСНОВЕ NSL-KDD")
    print("=" * 70)

    # 1. ЗАГРУЗКА ДАННЫХ 
    print("\n ШАГ 1: Загрузка данных")
    print("-" * 40)

    loader = NSLKDDDataLoader(data_path='data')

    # Загрузка обучающих и тестовых данных
    train_data = loader.load_data('train', use_20percent=False)
    test_data = loader.load_data('test')

    # Анализ данных
    print("\n Анализ обучающих данных:")
    loader.analyze_dataset(train_data)

    # 2. ПОДГОТОВКА ДАННЫХ
    print("\n\n ШАГ 2: Подготовка данных")
    print("-" * 40)

    # Выберите тип классификации (True - бинарная, False - многоклассовая)
    BINARY_CLASSIFICATION = False  # Измените на True для бинарной

    print(f"\nТип классификации: {'Бинарная (атака/нормальный)' if BINARY_CLASSIFICATION else 'Многоклассовая (5 категорий)'}")

    # Подготовка признаков и целевой переменной
    X_train, y_train, target_name = loader.prepare_target(
        train_data,
        binary=BINARY_CLASSIFICATION,
        filter_unknown=True
    )

    X_test, y_test, _ = loader.prepare_target(
        test_data,
        binary=BINARY_CLASSIFICATION,
        filter_unknown=True
    )

    print(f"\nРазмеры данных:")
    print(f"X_train: {X_train.shape}, y_train: {y_train.shape}")
    print(f"X_test: {X_test.shape}, y_test: {y_test.shape}")

    # Анализ распределения классов
    print(f"\n Распределение классов в обучающей выборке:")
    train_dist = pd.Series(y_train).value_counts()
    for class_name, count in train_dist.items():
        percentage = (count / len(y_train)) * 100
        print(f"  {class_name}: {count} записей ({percentage:.1f}%)")

    print(f"\n Распределение классов в тестовой выборке:")
    test_dist = pd.Series(y_test).value_counts()
    for class_name, count in test_dist.items():
        percentage = (count / len(y_test)) * 100
        print(f"  {class_name}: {count} записей ({percentage:.1f}%)")

    # Проверка совпадения классов
    train_classes = set(y_train.unique())
    test_classes = set(y_test.unique())

    print(f"\n Классы в обучающей выборке: {sorted(train_classes)}")
    print(f" Классы в тестовой выборке: {sorted(test_classes)}")

    # 3. ПРЕДОБРАБОТКА 
    print("\n\n ШАГ 3: Предобработка данных")
    print("-" * 40)

    # Определение типов признаков
    categorical_features, numerical_features = loader.get_feature_types(X_train)

    print(f"Категориальные признаки ({len(categorical_features)}): {categorical_features}")
    print(f"Числовые признаки ({len(numerical_features)}): первые 5 - {numerical_features[:5]}")

    # Создание и применение препроцессора
    preprocessor = DataPreprocessor(categorical_features, numerical_features)

    print("\nПреобразование данных...")

    # Обучаем на обучающих данных
    X_train_processed, y_train_encoded = preprocessor.fit_transform(X_train, y_train)

    # Преобразуем тестовые данные
    X_test_processed, y_test_encoded = preprocessor.transform(X_test, y_test)

    print(f" Размер после обработки:")
    print(f"  X_train: {X_train_processed.shape}")
    print(f"  X_test: {X_test_processed.shape}")

    # Анализ закодированных меток
    print(f"\n Классы после кодирования:")
    class_names = preprocessor.get_class_names()
    print(f"  Известные классы: {class_names}")

    # 4. ОБУЧЕНИЕ МОДЕЛЕЙ С РЕГУЛЯРИЗАЦИЕЙ 
    print("\n\n ШАГ 4: Обучение моделей")
    print("-" * 40)

    # Создаем словарь с параметрами моделей
    model_params = {
        'Decision Tree': {
            'max_depth': 10,  # Ограничиваем глубину
            'min_samples_split': 20,
            'min_samples_leaf': 10,
            'max_features': 'sqrt'
        },
        'Gradient Boosting': {
            'n_estimators': 100,
            'max_depth': 6,
            'learning_rate': 0.1,
            'min_samples_split': 20,
            'min_samples_leaf': 10,
            'subsample': 0.8,  # Случайные подвыборки
            'max_features': 'sqrt'
        },
        'KNN': {
        'n_neighbors': 50,
        'weights': 'distance',
        'metric': 'minkowski',
        'p': 2
        },
        'Random Forest': {
        'n_estimators': 200,
        'max_depth': 12,
        'min_samples_split': 30,
        'min_samples_leaf': 15,
        'max_features': 'sqrt',
        'bootstrap': True,
        'class_weight': 'balanced',
        'n_jobs': -1
        }
    }

    # Обучаем модели с параметрами против переобучения
    trainer = ModelTrainer(random_state=42)
    trainer.train_models(
        X_train_processed, y_train_encoded,
        X_test_processed, y_test_encoded,
        model_params=model_params  # Передаем параметры здесь
    )
    
    # 5. АНАЛИЗ РЕЗУЛЬТАТОВ
    print("\n\n ШАГ 5: Анализ результатов")
    print("-" * 40)

    # Сравнение моделей
    comparison_df = trainer.compare_models()
    print("\n Сравнение моделей:")
    print(comparison_df.to_string(index=False))

    # Анализ переобучения
    print("\n Анализ переобучения:")
    for _, row in comparison_df.iterrows():
        if 'Train CV (Acc)' in row and 'Test Accuracy' in row:
            train_val = row['Train CV (Acc)']
            test_val = row['Test Accuracy']
            
            # Безопасная проверка NaN
            train_val_float = float(train_val) if pd.notna(train_val) else 0
            test_val_float = float(test_val) if pd.notna(test_val) else 0
            
            if train_val_float > 0 and test_val_float > 0:
                diff = train_val_float - test_val_float
                if diff > 0.05:
                    print(f"  {row['Model']}: возможное переобучение (разница: {diff:.4f})")
                else:
                    print(f"  {row['Model']}: переобучение под контролем (разница: {diff:.4f})")

    # Лучшая модель
    best_model_name, best_result = trainer.get_best_model()
    print(f"\n Лучшая модель: {best_model_name}") 
    print(f"   Точность на тесте: {best_result['test_accuracy']:.4f}")

    # Детальный отчет лучшей модели
    y_pred_best = best_result['predictions']

    print(f"\n Детальный отчет ({best_model_name}):")

    # Используем inverse_transform для получения оригинальных меток
    y_test_original = preprocessor.inverse_transform(y_test_encoded)
    y_pred_original = preprocessor.inverse_transform(y_pred_best)

    # Создаем отчет
    report_dict = {}
    for class_name in class_names:
        if class_name in y_test_original:
            mask_true = y_test_original == class_name
            mask_pred = y_pred_original == class_name

            true_positives = np.sum((y_test_original == class_name) & (y_pred_original == class_name))
            false_positives = np.sum((y_test_original != class_name) & (y_pred_original == class_name))
            false_negatives = np.sum((y_test_original == class_name) & (y_pred_original != class_name))

            precision = true_positives / (true_positives + false_positives) if (true_positives + false_positives) > 0 else 0
            recall = true_positives / (true_positives + false_negatives) if (true_positives + false_negatives) > 0 else 0
            f1 = 2 * precision * recall / (precision + recall) if (precision + recall) > 0 else 0

            support = np.sum(mask_true)

            report_dict[class_name] = {
                'precision': precision,
                'recall': recall,
                'f1-score': f1,
                'support': support
            }

    # Добавляем средние значения
    precisions = [v['precision'] for v in report_dict.values()]
    recalls = [v['recall'] for v in report_dict.values()]
    f1_scores = [v['f1-score'] for v in report_dict.values()]
    supports = [v['support'] for v in report_dict.values()]

    report_dict['macro avg'] = {
        'precision': np.mean(precisions),
        'recall': np.mean(recalls),
        'f1-score': np.mean(f1_scores),
        'support': np.sum(supports)
    }

    report_dict['weighted avg'] = {
        'precision': np.average(precisions, weights=supports),
        'recall': np.average(recalls, weights=supports),
        'f1-score': np.average(f1_scores, weights=supports),
        'support': np.sum(supports)
    }

    report_df = pd.DataFrame(report_dict).transpose()
    print(report_df.round(3))

    # 6. ВИЗУАЛИЗАЦИЯ 
    print("\n\n ШАГ 6: Визуализация результатов")
    print("-" * 40)

    # Создание папки results если её нет
    os.makedirs('results', exist_ok=True)

    try:
        # 1. График сравнения моделей
        print("\n1. Создание графика сравнения моделей...")
        
        # Подготовка данных
        comparison_df['Test Accuracy'] = pd.to_numeric(comparison_df['Test Accuracy'], errors='coerce')
        if 'Train CV (Acc)' in comparison_df.columns:
            comparison_df['Train CV (Acc)'] = pd.to_numeric(comparison_df['Train CV (Acc)'], errors='coerce')
            train_col = 'Train CV (Acc)'
        else:
            train_col = None
        
        # Заполняем NaN
        comparison_df = comparison_df.fillna(0)
        
        plt.figure(figsize=(12, 8))
        
        models = comparison_df['Model']
        test_acc = comparison_df['Test Accuracy']
        
        x = np.arange(len(models))
        width = 0.6
        
        if train_col and train_col in comparison_df.columns:
            train_acc = comparison_df[train_col]
            width = 0.35
            
            plt.bar(x - width/2, train_acc, width, label='Train CV', alpha=0.8, color='steelblue')
            plt.bar(x + width/2, test_acc, width, label='Test', alpha=0.8, color='lightcoral')
            plt.legend()
        else:
            plt.bar(x, test_acc, width, alpha=0.8, color='lightcoral')
        
        plt.xlabel('Модели', fontsize=12)
        plt.ylabel('Accuracy', fontsize=12)
        plt.title('Сравнение точности моделей', fontsize=14, fontweight='bold')
        plt.xticks(x, models, rotation=45, ha='right')
        plt.grid(True, alpha=0.3, axis='y')
        
        # Добавление значений - БЕЗОПАСНЫЙ СПОСОБ
        for i, val in enumerate(test_acc):
            try:
                val_float = float(val)
                if val_float > 0:
                    plt.text(i if train_col is None else i + width/2, 
                            val_float + 0.01, f'{val_float:.3f}', 
                            ha='center', va='bottom', fontsize=9)
            except (ValueError, TypeError):
                continue
        
        if train_col and train_col in comparison_df.columns:
            for i, val in enumerate(train_acc):
                try:
                    val_float = float(val)
                    if val_float > 0:
                        plt.text(i - width/2, val_float + 0.01, f'{val_float:.3f}', 
                                ha='center', va='bottom', fontsize=9)
                except (ValueError, TypeError):
                    continue
        
        plt.tight_layout()
        plt.savefig('results/model_comparison.png', dpi=300, bbox_inches='tight')
        plt.show()
        print(" График сохранен: results/model_comparison.png")
        
    except Exception as e:
        print(f" Ошибка при создании графика сравнения: {e}")

    try:
        # 2. Распределение классов 
        print("\n2. График распределения классов...")
        
        plt.figure(figsize=(10, 6))
        
        if not train_dist.empty and len(train_dist) > 0:
            colors = plt.cm.Set3(np.linspace(0, 1, len(train_dist)))
            train_dist.plot(kind='bar', color=colors)
            
            plt.title('Распределение классов в обучающей выборке', fontsize=14, fontweight='bold')
            plt.xlabel('Класс', fontsize=12)
            plt.ylabel('Количество записей', fontsize=12)
            plt.xticks(rotation=45)
            plt.grid(True, alpha=0.3, axis='y')
            
            # Добавление значений
            for i, v in enumerate(train_dist.values):
                if not pd.isna(v):
                    plt.text(i, v + max(train_dist.values)*0.01, str(int(v)),
                            ha='center', va='bottom', fontsize=10)
            
            plt.tight_layout()
            plt.savefig('results/class_distribution.png', dpi=300, bbox_inches='tight')
            plt.show()
            print(" График сохранен: results/class_distribution.png")
        else:
            print(" Нет данных для графика распределения классов")
            
    except Exception as e:
        print(f" Ошибка при создании графика распределения: {e}")

    try:
        # 3. Матрица ошибок 
        print("\n3. Матрица ошибок для лучшей модели...")
        
        if y_pred_best is not None and len(y_pred_best) > 0:
            # Создаем матрицу ошибок
            cm = confusion_matrix(y_test_encoded, y_pred_best)
            
            plt.figure(figsize=(10, 8))
            sns.heatmap(cm, annot=True, fmt='d', cmap='Blues',
                       xticklabels=class_names,
                       yticklabels=class_names)
            plt.title(f'Матрица ошибок: {best_model_name}', fontsize=14, fontweight='bold')
            plt.ylabel('Истинные значения', fontsize=12)
            plt.xlabel('Предсказанные значения', fontsize=12)
            plt.tight_layout()
            plt.savefig('results/confusion_matrix.png', dpi=300, bbox_inches='tight')
            plt.show()
            print(f" Матрица ошибок сохранена: results/confusion_matrix.png")
        else:
            print(" Нет предсказаний для лучшей модели")
            
    except Exception as e:
        print(f" Ошибка при создании матрицы ошибок: {e}")

    # 7. ДОПОЛНИТЕЛЬНЫЙ АНАЛИЗ 
    print("\n\n ШАГ 7: Дополнительный анализ")
    print("-" * 40)

    # Анализ важности признаков
    try:
        if best_result['model'] is not None and hasattr(best_result['model'], 'feature_importances_'):
            print("\n Анализ важности признаков:")

            feature_names = preprocessor.get_feature_names()
            importances = best_result['model'].feature_importances_

            # Создание DataFrame с важностью признаков
            feature_importance_df = pd.DataFrame({
                'Feature': feature_names,
                'Importance': importances
            }).sort_values('Importance', ascending=False)

            print("\nТоп-10 важных признаков:")
            print(feature_importance_df.head(10).to_string(index=False))

            # Визуализация топ-15 признаков
            plt.figure(figsize=(12, 8))
            top_features = feature_importance_df.head(15)

            plt.barh(range(len(top_features)), top_features['Importance'])
            plt.yticks(range(len(top_features)), top_features['Feature'], fontsize=9)
            plt.xlabel('Важность признака', fontsize=12)
            plt.title('Топ-15 важных признаков для классификации', fontsize=14, fontweight='bold')
            plt.gca().invert_yaxis()
            plt.tight_layout()
            plt.savefig('results/feature_importance.png', dpi=300, bbox_inches='tight')
            plt.show()
            print(" График важности признаков сохранен")
        else:
            print(" Модель не поддерживает feature_importances_")
    except Exception as e:
        print(f" Ошибка при анализе важности признаков: {e}")

    # Анализ ошибок классификации
    try:
        print("\n Анализ ошибок классификации по классам:")
        error_analysis = pd.DataFrame({
            'True': y_test_original,
            'Predicted': y_pred_original,
            'Correct': y_test_original == y_pred_original
        })

        error_by_class = error_analysis.groupby('True')['Correct'].agg(['mean', 'count'])
        error_by_class['error_rate'] = (1 - error_by_class['mean']) * 100
        error_by_class['error_count'] = error_by_class['count'] * (1 - error_by_class['mean'])

        print(error_by_class[['error_rate', 'error_count']].round(2))
    except Exception as e:
        print(f" Ошибка при анализе ошибок: {e}")

    # 8. ИТОГИ 
    print("\n\n" + "=" * 70)
    print(" ИТОГОВЫЙ ОТЧЕТ")
    print("=" * 70)

    print(f"\n Общая статистика:")
    print(f"  • Всего записей: {len(train_data) + len(test_data)}")
    print(f"  • Обучающая выборка: {len(X_train)} записей")
    print(f"  • Тестовая выборка: {len(y_test_encoded)} записей")
    print(f"  • Количество классов: {len(class_names)}")
    print(f"  • Классы: {', '.join(class_names)}")

    print(f"\n Результаты:")
    print(f"  • Лучшая модель: {best_model_name}")
    print(f"  • Точность на тесте: {best_result['test_accuracy']:.4f}")
    
    if best_result.get('cv_mean') is not None:
        diff = abs(best_result['cv_mean'] - best_result['test_accuracy'])
        print(f"  • Разница CV/Test: {diff:.4f}")
        if diff > 0.05:
            print(f"  ⚠ Внимание: возможное переобучение (разница > 0.05)")
        else:
            print(f"  ✓ Переобучение под контролем")

    print(f"\n Распределение классов:")
    for class_name in class_names:
        count_train = (y_train == class_name).sum()
        count_test = (y_test_original == class_name).sum()
        percentage_train = (count_train / len(y_train)) * 100 if len(y_train) > 0 else 0
        percentage_test = (count_test / len(y_test_original)) * 100 if len(y_test_original) > 0 else 0
        print(f"  • {class_name}: обучающих={count_train} ({percentage_train:.1f}%), тестовых={count_test} ({percentage_test:.1f}%)")

    print(f"\n Выводы:")
    print("  1. Модели машинного обучения эффективны для классификации сетевого трафика")
    print("  2. Регуляризация помогает контролировать переобучение")
    print("  3. Наибольшие трудности с редкими классами (R2L, U2R)")
    print("  4. Важными являются признаки, описывающие статистику соединений")

    print(f"\n Рекомендации для уменьшения переобучения:")
    print("  1. Использовать кросс-валидацию при выборе гиперпараметров")
    print("  2. Применять балансировку классов (SMOTE, ADASYN)")
    print("  3. Использовать регуляризацию (ограничение глубины деревьев)")
    print("  4. Добавить dropout в ансамблевые методы")
    print("  5. Увеличить объем обучающих данных")

    # Сохранение результатов
    print(f"\n Сохранение результатов...")
    
    try:
        # Сохранение сравнения моделей
        comparison_df.to_csv('results/model_comparison.csv', index=False)
        
        # Сохранение детального отчета
        report_df.to_csv('results/detailed_report.csv')
        
        # Сохранение распределения классов
        pd.Series(y_train).value_counts().to_csv('results/class_distribution.csv')
        
        # Сохранение конфигурации
        with open('results/config.txt', 'w', encoding='utf-8') as f:
            f.write(f"Датасет: NSL-KDD\n")
            f.write(f"Тип классификации: {'Бинарная' if BINARY_CLASSIFICATION else 'Многоклассовая'}\n")
            f.write(f"Лучшая модель: {best_model_name}\n")
            f.write(f"Точность: {best_result['test_accuracy']:.4f}\n")
            f.write(f"Классы: {', '.join(class_names)}\n")
        
        print(" Результаты сохранены в папке 'results/'")
    except Exception as e:
        print(f" Ошибка при сохранении файлов: {e}")

    print("\n" + "=" * 70)
    print(" Анализ завершен успешно!")
    print("=" * 70)

if __name__ == "__main__":
    main()
