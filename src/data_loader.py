# Модуль для загрузки и первичной обработки данных NSL-KDD
import pandas as pd
import numpy as np
import os

# Класс для работы с данными NSL-KDD
class NSLKDDDataLoader:

    def __init__(self, data_path='data'):
        self.data_path = data_path 
        # список имен всех 43 колонок
        self.features = [ 
            "duration", "protocol_type", "service", "flag", "src_bytes", "dst_bytes",
            "land", "wrong_fragment", "urgent", "hot", "num_failed_logins",
            "logged_in", "num_compromised", "root_shell", "su_attempted", "num_root",
            "num_file_creations", "num_shells", "num_access_files", "num_outbound_cmds",
            "is_host_login", "is_guest_login", "count", "srv_count", "serror_rate",
            "srv_serror_rate", "rerror_rate", "srv_rerror_rate", "same_srv_rate",
            "diff_srv_rate", "srv_diff_host_rate", "dst_host_count", "dst_host_srv_count",
            "dst_host_same_srv_rate", "dst_host_diff_srv_rate", "dst_host_same_src_port_rate",
            "dst_host_srv_diff_host_rate", "dst_host_serror_rate", "dst_host_srv_serror_rate",
            "dst_host_rerror_rate", "dst_host_srv_rerror_rate", "label", "difficulty"
        ]

        # Словарь для группировки атак по категориям
        self.attack_categories = {
            'DoS': [
                "back", "land", "neptune", "pod", "smurf", "teardrop",
                "mailbomb", "apache2", "processtable", "udpstorm"
            ],
            'Probe': [
                "satan", "ipsweep", "nmap", "portsweep", "mscan", "saint"
            ],
            'R2L': [
                "guess_passwd", "ftp_write", "imap", "phf", "multihop",
                "warezmaster", "warezclient", "spy", "xlock", "xsnoop",
                "snmpguess", "snmpgetattack", "httptunnel", "sendmail", "named"
            ],
            'U2R': [
                "buffer_overflow", "loadmodule", "rootkit", "perl",
                "sqlattack", "xterm", "ps"
            ]
        }

    # Загрузка данных из файла
    def load_data(self, dataset='train', use_20percent=False):
        # 1. определение имени файла
        if dataset == 'train':
            if use_20percent:
                filename = 'KDDTrain+_20Percent.txt' # использование маленькой версии
            else:
                filename = 'KDDTrain+.txt' # использование полной версии
        elif dataset == 'test':
            filename = 'KDDTest+.txt'
        else:
            raise ValueError("dataset должен быть 'train' или 'test'") # проверка на ошибку
        
        # 2. создание полного пути к файлу
        filepath = os.path.join(self.data_path, filename)

        if not os.path.exists(filepath):
            raise FileNotFoundError("Файл не найден:", filepath) # проверка на ошибку

        print("Загрузка", filename)

        # 3. Чтение CSV файла без заголовков, присваивая имена колонкам
        data = pd.read_csv(filepath, names=self.features) # используем список из 43 имен колонок, так как в файлах нет заголовков
        print(f"Загружено {len(data)} записей")

        return data

    # Преобразует метку атаки в категорию
    def categorize_attack(self, label):
        
        if label == "normal":
            return "normal" 

        # Проверка каждой категории атак
        for category, attacks in self.attack_categories.items():
            if label in attacks:
                return category

        # Если атака не в списке - вернуть как 'other'
        return "other"

    # Подготовка целевой переменной
    def prepare_target(self, data, binary=False, filter_unknown=False):
        # 1. Создание копии данных, чтобы не испортился оригинал
        df = data.copy()

        # 2. Создание целевой переменной
        if binary:
            # Бинарная классификация: normal vs attack
            df['target'] = df['label'].apply(
                lambda x: 0 if x == 'normal' else 1
            )
            target_name = 'attack'
        else:
            # Многоклассовая классификация
            df['target'] = df['label'].apply(self.categorize_attack)
            target_name = 'attack_category'

        # 3. Фильтрация записей с меткой 'other' если нужно
        if filter_unknown:
            initial_count = len(df) 
            df = df[df['target'] != 'other']
            filtered_count = initial_count - len(df) # количество удаленных
            if filtered_count > 0:
                print(f"Отфильтровано {filtered_count} записей с меткой 'other'")

        # 4. Удаление ненужных столбцов
        if 'difficulty' in df.columns:
            df = df.drop(columns=['difficulty'])

        # 5. Разделение на признаки(Х) и целевую переменную (у)
        X = df.drop(columns=['label', 'target'])
        y = df['target']

        return X, y, target_name # возвращаем признаки, метки и название задачи
    
    # Определяет типы признаков
    def get_feature_types(self, X):
        
        categorical_features = ["protocol_type", "service", "flag"]
        numerical_features = [col for col in X.columns if col not in categorical_features]

        return categorical_features, numerical_features
    
    # Выводит анализ датасета
    def analyze_dataset(self, data):

        print("=" * 60)
        print("АНАЛИЗ ДАТАСЕТА")
        print("=" * 60)

        print(f"Размер: {data.shape}")
        print(f"Количество признаков: {len(data.columns)}")
        print(f"Количество записей: {len(data)}")

        # Анализ пропущенных значений
        missing = data.isnull().sum()
        if missing.sum() > 0:
            print(f"\nПропущенные значения:")
            print(missing[missing > 0])
        else:
            print("\nПропущенных значений нет")

        # Анализ типов данных
        print(f"\nТипы данных:")
        print(data.dtypes.value_counts())

        # Распределение меток
        if 'label' in data.columns:
            print(f"\nРаспределение меток:")
            label_counts = data['label'].value_counts()
            print(f"Всего уникальных меток: {len(label_counts)}")
            print(f"\nТоп-10 меток:")
            print(label_counts.head(10))

            # Проверка на наличие 'other' меток после категоризации
            if 'target' in data.columns:
                target_counts = data['target'].value_counts()
                if 'other' in target_counts.index:
                    print(f"\nОбнаружены записи с меткой 'other': {target_counts['other']}")