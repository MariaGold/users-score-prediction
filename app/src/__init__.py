from flask import Flask, request, jsonify, make_response

from Modules.Model import Model
from Modules.File import File

app = Flask(__name__)

# Добавление данных в модель
@app.route('/add_data', methods=['PUT']) 
def add_data ():
    if 'file' not in request.files:
        return make_response("Отсутствует файл с данными", 400)
    else:
        file = File(request.files['file'])
    
        if (file.is_allowed()):
            model = Model()
            data = file.get_data()
            result = model.add_train_data(file, data)

            if result:
                return make_response('Данные успешно добавлены', 200)
            else:
                return make_response('Не удалось обновить обучающую выборку', 403)
        else:
            return make_response("Недопустимый формат файла", 400)


# Обучение модели по всем к данному времени загруженным данным
@app.route('/retrain', methods=['PUT']) 
def retrain ():
    model = Model()
    experiment_id = model.train()
    
    if experiment_id:
        return jsonify({'experiment_id': experiment_id})
    else:
        return make_response('Не удалось обучить модель', 403)


# Получение метрик эксперимента с определенным номером
@app.route('/metrics/<experiment_id>', methods=['GET']) 
def metrics (experiment_id):
    model = Model()
    metrics = model.get_metrics_by_experiment(experiment_id)
    
    if metrics:
        return jsonify(metrics)
    else:
        return make_response('Не удалось получить метрики', 403)


# Переключение моделей
@app.route('/deploy/<experiment_id>', methods=['POST']) 
def deploy (experiment_id):
    model = Model()
    switched_id = model.switch(experiment_id)
    
    if switched_id:
        return make_response(f'Основная модель переключена на модель номер {switched_id}', 200) 
    else:
        return make_response('Не удалось переключить модель', 403) 


# Получение метаданных актуальной модели
@app.route('/metadata', methods=['GET']) 
def metadata ():
    model = Model()
    metadata = model.get_model_metadata()
    
    if metadata:
        return jsonify(metadata)
    else:
        return make_response('Не удалось получить метаданные модели', 403)


# Получение ответа для тестового объекта
@app.route('/forward', methods=['POST']) 
def forward ():
    if request.headers.get('Content-Type') == 'application/json':
        data = request.get_json()
        
        model = Model()
        result = model.predict(data)
        
        if result is not None:
            return jsonify(result)
        else:
            return make_response('Модель не смогла обработать данные', 403)
    else:
        return make_response("bad request", 400)


# Получение ответа для тестовой выборки
@app.route('/forward_batch', methods=['POST']) 
def forward_batch ():
    if 'file' not in request.files:
        return make_response("Отсутствует файл с данными", 400)
    else:
        file = File(request.files['file'])
        
        if (file.is_allowed()):
            data = file.get_data()
            model = Model()
            result = model.predict(data)

            if result is not None:
                return jsonify(result)
            else:
                return make_response('Модель не смогла обработать данные', 403)
        else:
            return make_response("Недопустимый формат файла", 400)


# Получение метрик для тестовой выборки
@app.route('/evaluate', methods=['GET']) 
def evaluate ():
    if 'file' not in request.files:
        return make_response("Отсутствует файл с данными", 400)
    else:
        file = File(request.files['file'])
        
        if (file.is_allowed()):
            data = file.get_data()
            model = Model()
            result = model.predict(data)
            metrics = model.metrics

            if result is not None:
                return jsonify(metrics)
            else:
                return make_response('Модель не смогла обработать данные', 403)
        else:
            return make_response("Недопустимый формат файла", 400)


if __name__ == '__main__':
    app.run(host='0.0.0.0', debug=False)