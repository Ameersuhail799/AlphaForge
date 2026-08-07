import pandas as pd

from config.settings import MODEL_COMPARISON_REPORT_PATH
from tests.utils import load_test_data

from src.dataset.builder import DatasetBuilder
from src.features.feature_pipeline import FeaturePipeline
from src.models.evaluator import Evaluator
from src.models.predictor import Predictor
from src.models.registry import ModelRegistry
from src.models.trainer import Trainer


def main():

    df = load_test_data()
    engineered_df = FeaturePipeline().generate(df)
    bundle = DatasetBuilder(scale=True).build(
        engineered_df,
        symbol="RELIANCE.NS",
    )

    registry = ModelRegistry()
    assert "random_forest" in registry.list_models()

    model = registry.create("random_forest")
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

    experiment_id = "random_forest_integration_test"
    model_path = trainer.save_model(trained_model, experiment_id)
    importance_path = trainer.export_feature_importance(
        trained_model,
        experiment_id,
        file_name="random_forest_feature_importance.csv",
    )
    experiment_path = trainer.save_experiment(
        trained_model,
        bundle,
        metrics,
        experiment_id,
        prediction_time_seconds=predictor.prediction_time_seconds,
    )
    comparison = pd.read_csv(MODEL_COMPARISON_REPORT_PATH)

    assert len(predictions) == len(bundle.X_test)
    assert len(probabilities) == len(bundle.X_test)
    assert probabilities.between(0, 1).all()
    assert metrics.accuracy >= 0
    assert model_path.exists()
    assert importance_path.exists()
    assert experiment_path.exists()
    assert MODEL_COMPARISON_REPORT_PATH.exists()
    assert "random_forest" in comparison["Model"].to_list()
    assert importance_path.name == "random_forest_feature_importance.csv"

    print()
    print("=" * 60)
    print("RANDOM FOREST SUCCESS")
    print("=" * 60)
    print()
    print(metrics.to_dict())
    print(trained_model.feature_importance().head(10))


if __name__ == "__main__":
    main()
