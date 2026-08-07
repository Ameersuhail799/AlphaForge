from tests.utils import load_test_data

from src.dataset.builder import DatasetBuilder
from src.features.feature_pipeline import FeaturePipeline
from src.models.evaluator import Evaluator
from src.models.logistic_regression import LogisticRegressionModel
from src.models.predictor import Predictor
from src.models.trainer import Trainer


def main():

    df = load_test_data()

    engineered_df = FeaturePipeline().generate(df)
    bundle = DatasetBuilder(scale=True).build(
        engineered_df,
        symbol="RELIANCE.NS",
    )

    model = LogisticRegressionModel()
    trainer = Trainer()
    trained_model = trainer.train(model, bundle)

    predictor = Predictor()
    predictions = predictor.predict(trained_model, bundle)
    probabilities = predictor.predict_probabilities(trained_model, bundle)

    metrics = Evaluator().evaluate(
        bundle,
        predictions.to_numpy(),
        probabilities.to_numpy(),
    )

    experiment_id = "logistic_regression_integration_test"
    model_path = trainer.save_model(trained_model, experiment_id)
    importance_path = trainer.export_feature_importance(
        trained_model,
        experiment_id,
    )
    experiment_path = trainer.save_experiment(
        trained_model,
        bundle,
        metrics,
        experiment_id,
    )
    loaded_model = trainer.load_model(model_path)

    assert len(predictions) == len(bundle.X_test)
    assert len(probabilities) == len(bundle.X_test)
    assert probabilities.between(0, 1).all()
    assert metrics.accuracy >= 0
    assert metrics.precision >= 0
    assert metrics.recall >= 0
    assert metrics.f1 >= 0
    assert len(metrics.confusion_matrix) == 2
    assert model_path.exists()
    assert importance_path.exists()
    assert experiment_path.exists()
    assert loaded_model.model_name == trained_model.model_name

    print()
    print("=" * 60)
    print("LOGISTIC REGRESSION SUCCESS")
    print("=" * 60)
    print()
    print(metrics.to_dict())


if __name__ == "__main__":
    main()
