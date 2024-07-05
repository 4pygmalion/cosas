import mlflow

TRACKING_URI = "http://219.252.39.224:5000/"
EXP_NAME = "cosas"


def get_experiment(experiment_name=EXP_NAME):
    mlflow.set_tracking_uri(TRACKING_URI)

    client = mlflow.tracking.MlflowClient(TRACKING_URI)
    experiment = client.get_experiment_by_name(experiment_name)

    if not experiment:
        client.create_experiment(experiment_name)
        return client.get_experiment_by_name(experiment_name)

    return experiment
