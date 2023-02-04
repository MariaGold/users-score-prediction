from Modules.File import *
from datetime import datetime

from sklearn.preprocessing import StandardScaler, OneHotEncoder
from sklearn.compose import ColumnTransformer 
from sklearn.pipeline import Pipeline
from sklearn.tree import DecisionTreeRegressor
from sklearn.metrics import mean_squared_error, mean_absolute_error

from onnxruntime import InferenceSession
from skl2onnx.common.data_types import FloatTensorType, StringTensorType, Int64TensorType
from mlprodict.onnx_conv import guess_schema_from_data
from skl2onnx import convert_sklearn

import numpy as np
import pandas as pd
from flask import jsonify
import os, shutil

class Model():
    def __init__(self):
        self.fname = 'data.csv'
        self.mdir = './data'
        self.train_datasets_dir = './data/datasets'
        self.train_data_path = './data/data.csv'
        self.onnx_models_dir = './data/experiments'
        self.onnx_current_model_path = './data/model.onnx'

        self.categorical_features = ['game_mode', 'map']
        self.numerical_features = ['plrs_share', 'plrs_num', 'btls_n_total', 
                      'max_rating_diff', 'mean_rating_diff', 
                      'avg_kd_season_diff', 'avg_dmg_season_diff', 'avg_btls_n_season_diff',
                       'avg_kd_season_game_mode_diff', 'avg_dmg_season_game_mode_diff', 'avg_btls_n_game_mode_diff',
                      'avg_kd_season_map_diff', 'avg_dmg_season_map_diff', 'avg_btls_n_season_map_diff']

        self.model = None
        self.metrics = dict()
        self.train_metrics = dict()


    def train(self):
        df = None

        if os.path.exists(self.train_data_path):
            df = pd.read_csv(self.train_data_path)
        else:
            return False

        # Сортируем данные по времени
        df = df.sort_values('event_timestamp')
         # Пропуски вещественных значений заменяе нулями, а категориальных - 'undefined'
        df[self.numerical_features] = df[self.numerical_features].fillna(0)
        df[self.categorical_features] = df[self.categorical_features].fillna('undefined')
        # Выбираем размер обучающей выборки
        train_size = int(df.shape[0] * 0.8)
        # Обучающая выборка
        X_train = df.iloc[:train_size, :][self.categorical_features + self.numerical_features]
        y_train = abs(df.iloc[:train_size, :]['k'])
        # Тестовая выборка
        X_test = df.iloc[train_size:, :][self.categorical_features + self.numerical_features]
        y_test = abs(df.iloc[train_size:, :]['k'])
        # Обучение модели
        self.fit(X_train, y_train, X_test, y_test)

        experiment_id = self.get_next_experiment_id() - 1

        return experiment_id


    def fit(self, X_train, y_train, X_test, y_test) -> None:
        # Создаем трансформер признаков
        categorical_transformer = Pipeline(steps=[ ('ohe', OneHotEncoder(handle_unknown="ignore")) ])
        numerical_transformer = Pipeline(steps=[ ('scaler', StandardScaler()) ])

        preprocessor = ColumnTransformer(transformers=[
                                        ('cat', categorical_transformer, self.categorical_features),
                                        ('num', numerical_transformer, self.numerical_features),
                                    ])

        # Создаем пайплайн
        self.model = Pipeline(steps=[
            ('preprocessor', preprocessor),
            ('regression', DecisionTreeRegressor(max_depth=7))
        ])

        # Обучаем модель
        self.model.fit(X_train, y_train)

        # Предсказание по тестовой выборке 
        y_pred = self.model.predict(X_test)

        # Метрики по тестовой выборке
        self.train_metrics['MSE'] = mean_squared_error(y_test, y_pred)
        self.train_metrics['MAE'] = mean_absolute_error(y_test, y_pred)

        # Сохранение модели в onnx-формат
        self.save2onnx(X_train)


    def save2onnx(self, df):
        for column in df.columns:
            if column not in self.categorical_features:
                df[column] = df[column].astype(np.float32)

        onx = convert_sklearn(self.model, initial_types=guess_schema_from_data(df))

        meta = onx.metadata_props.add()
        meta.key='commit'
        meta.value=open('/app/data/commit.txt','r').readline()

        meta = onx.metadata_props.add()
        meta.key='model_date'
        meta.value=str(datetime.now())

        meta = onx.metadata_props.add()
        meta.key='model_experiment_id'
        experiment_id = self.get_next_experiment_id()
        meta.value=str(experiment_id)

        meta = onx.metadata_props.add()
        meta.key='model_features_num'
        meta.value=str(df.shape[1])

        meta = onx.metadata_props.add()
        meta.key='model_train_metrics'
        meta.value=str(self.train_metrics)

        # Сохраняем экспримент под своим номером
        with open(f'{self.onnx_models_dir}/model_{experiment_id}.onnx', "wb") as f:
            f.write(onx.SerializeToString())

        # Обновляем текущую модель
        if os.path.exists(self.onnx_current_model_path):
            os.remove(self.onnx_current_model_path)
        with open(self.onnx_current_model_path, "wb") as f:
            f.write(onx.SerializeToString())


    def predict(self, data):
        X = None
        y = None

        if isinstance(data, dict):
            values = [[data[key] for key in data]]
            keys = [key for key in data]
            X = pd.DataFrame(data=values, columns=keys)
        elif isinstance(data, pd.DataFrame):
            X = data
        else:
            return None

        if 'k' in X.columns:
            y = X['k']
            X = X.drop(['Unnamed: 0', 'k'], axis=1)

        sess = InferenceSession(self.onnx_current_model_path)

        for column in X.columns:
            if column not in self.categorical_features:
                X[column] = X[column].astype(np.float32)
        
        inputs = {column: X[column].values.reshape((-1, 1))
          for column in X.columns}

        if X.shape[1] != int(sess.get_modelmeta().custom_metadata_map['model_features_num']):
            return None

        res = sess.run(None, inputs)

        y_pred = res[0]

        if y is not None:
            self.metrics['MSE'] = mean_squared_error(y, y_pred)
            self.metrics['MAE'] = mean_absolute_error(y, y_pred)

        return y_pred


    def get_model_metadata(self):
        if os.path.exists(self.onnx_current_model_path):
            sess = InferenceSession(self.onnx_current_model_path)
            meta = sess.get_modelmeta()

            return meta.custom_metadata_map
        else:
            return False


    def get_next_experiment_id(self):
        idx = 0

        for m in os.listdir(self.onnx_models_dir):
            idx = max(idx, int(m[6:str.find(m, '.')]))

        return idx + 1

    def get_next_dataset_id(self):
        idx = 0

        for d in os.listdir(self.train_datasets_dir):
            idx = max(idx, int(d[5:str.find(d, '.')]))

        return idx + 1

    def add_train_data(self, file: File, df_to_add):
        dataset_id = self.get_next_dataset_id()
        file.save(f'{self.train_datasets_dir}/data_{dataset_id}.{file.extension}')

        df = None

        if os.path.exists(self.train_data_path):
            df = pd.read_csv(self.train_data_path)

            try:
                df = pd.concat([df, df_to_add.reset_index(drop=True)], axis=0)
            except:
                return False
        else:
            df = df_to_add.copy()

        df.to_csv(self.train_data_path)

        return True


    def get_metrics_by_experiment(self, experiment_id):
        path = f'{self.onnx_models_dir}/model_{experiment_id}.onnx'

        if os.path.exists(path):
            sess = InferenceSession(path)
            return sess.get_modelmeta().custom_metadata_map['model_train_metrics']
        else:
            return False


    def switch(self, model_id):
        target_model_path = os.path.join(self.onnx_models_dir, f'model_{model_id}.onnx')

        if os.path.exists(target_model_path):
            shutil.copy2(target_model_path, self.onnx_current_model_path)
        else:
            return None

        return model_id