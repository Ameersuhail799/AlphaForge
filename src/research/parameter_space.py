"""Hyperparameter search spaces for AlphaForge research."""

from __future__ import annotations


class ParameterSpace:
    """Define reusable model parameter spaces for optimization."""

    RANDOM_FOREST = {
        "n_estimators": [100, 200, 300, 500],
        "max_depth": [5, 10, 20, None],
        "min_samples_leaf": [1, 2, 5, 10],
        "max_features": ["sqrt", "log2", None],
    }

    @classmethod
    def get(cls, model_name: str) -> dict[str, list[object]]:
        """Return the parameter space for a registered model.

        Args:
            model_name: Registered AlphaForge model name.

        Returns:
            Search space for the model.

        Raises:
            ValueError: If the model has no supported search space.
        """

        if model_name == "random_forest":
            return cls.RANDOM_FOREST

        raise ValueError(f"No parameter space is defined for: {model_name}")