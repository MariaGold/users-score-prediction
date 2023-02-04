# users-score-prediction

Devops project with basic ML-pipeline

## Description
The project is devoted to prediction of the difference between teams' scores in a mobile shooter. The project demonstrates basic features of ML-pipeline which are typically implemented in such a kind of projects. All the models are stored in the ```onnx```-format. The app has the following methods:

### /add_data
Adds more samples for the model to be learned. Data must be compatible with the current dataset. Otherwise, it won't be uploaded.

### /retrain
Retrains a model (commonly after adding more data). This method returns the identificator of the experiment ```experiment_id```.

### /metrics/<experiment_id>
Returns the metrics related to a particular experiment by its ```experiment_id```.

### /deploy/<experiment_id>
Switches the currently used model by another one by its ```experiment_id```.

### /metadata
Returns metadata of the currently used model. Metadata include ```commit```, ```model_date```, ```model_experiment_id```, ```model_features_num```, ```model_train_metrics```.

### /forward
Applies the currently used model to one object.

### /forward_batch
Applies the currently used model to a batch of objects.

### /evaluate
Returns metrics for test samples.

## Methods

## Run the project:
From the root directory run the following command
```
docker-compose up --build
```
