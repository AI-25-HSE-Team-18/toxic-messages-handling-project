# Запуск api-инференса

## 1. Модель по умолчанию
<!-- ### Выбор конфига

```bash
export MODEL_CONFIG="config.json" 
``` -->

### Проверка конфига

```json

    "aws_endpoint_url": "https://storage.yandexcloud.net", # (optional)
    "aws_access_key_id": "your_access_key_id", # (optional)
    "aws_secret_access_key": "your_secret_access_key", # (optional)

    "predictors": [
        {
            "storage_type": "local/s3", # flag local/s3 model 
            "bucket_name": "bucket_name", # ignored if local
            "model_path": "model_path.pkl", # local / s3 path without bucket
            "encoder_path": "", # optional, for classic ml models only 
            "additional_data_path": "", # optional, whether some additional data is used
            "description": "optional", # info to identify model version
        }
    ...
    ]
```

### Сборка и запуск
```bash
docker build -t toxicity-api .
docker run -p 8000:8000 toxicity-api 
```
Готово!

## 2. Как заменить модель по умолчанию/препроцессинг

### Добавление модели как класса-наследника от Model
`/src/services/model.py`

```python
# если добавляется не pickle+classic ml модель, определите load_custom_weights()
# (с функциями для локальной / s3 загрузки)
@abstractmethod 
def load_custom_weights(self, model_path: str):
    """Method to load any of non-pickle models, e.g. from torch, tensorflow, etc."""
    
    self._model_weights = None

# текст -> входные векторы модели
@abstractmethod
def preprocess(self, text: str):
    """text -> model input array"""
    pass

# предикт на загруженных весах 
@abstractmethod
def predict(self, inputs):
    """Do prediction based on input and return pred"""
    pass
```

### Импорт и использование в forward()
TODO: multiprocessing + более умная замена в forward
`src/routers/forward.py`
```python
...
model = LinearSVMModel()
...
```
### Сборка и запуск
```bash
docker build -t toxicity-api .
docker run -p 8000:8000 toxicity-api 
```
Готово!