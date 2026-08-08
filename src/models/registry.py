"""Central model registry for AlphaForge machine learning models."""

from __future__ import annotations

from collections.abc import Callable

from src.models.base_model import BaseModel
from src.models.logistic_regression import LogisticRegressionModel
from src.models.random_forest import RandomForestModel
from src.models.xgboost_model import XGBoostModel
from src.utils.logger import get_logger

logger = get_logger(__name__)

ModelFactory = Callable[..., BaseModel]


class ModelRegistry:
    """Register, construct, and list AlphaForge machine learning models."""

    def __init__(self) -> None:
        """Initialize the registry with production model factories."""

        self._models: dict[str, ModelFactory] = {}
        self.register("logistic_regression", LogisticRegressionModel)
        self.register("random_forest", RandomForestModel)
        self.register("xgboost", XGBoostModel)

    def register(
        self,
        name: str,
        model_factory: ModelFactory,
    ) -> None:
        """Register a model factory under a unique name.

        Args:
            name: Unique model identifier.
            model_factory: Callable that constructs a model instance.

        Raises:
            ValueError: If the model name is already registered.
        """

        if name in self._models:
            raise ValueError(f"Model already registered: {name}")

        self._models[name] = model_factory

        logger.info("Registered model '%s'.", name)

    def get_model(self, name: str) -> ModelFactory:
        """Retrieve a registered model factory by name.

        Args:
            name: Registered model identifier.

        Returns:
            Model factory associated with the name.

        Raises:
            ValueError: If the model name is not registered.
        """

        model_factory = self._models.get(name)

        if model_factory is None:
            raise ValueError(f"Unknown model: {name}")

        return model_factory

    def create(self, name: str, **parameters: object) -> BaseModel:
        """Create a registered model instance.

        Args:
            name: Registered model identifier.
            **parameters: Model constructor parameters.

        Returns:
            New model instance.
        """

        logger.info("Creating model '%s' from registry.", name)

        return self.get_model(name)(**parameters)

    def list_models(self) -> list[str]:
        """List registered model names in alphabetical order.

        Returns:
            Available model names.
        """

        return sorted(self._models)
