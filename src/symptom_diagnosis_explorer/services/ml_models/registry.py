"""Registry for model development services supporting multiple ML frameworks."""

from typing import Callable, Type

from symptom_diagnosis_explorer.models.model_development import ClassificationConfig
from symptom_diagnosis_explorer.services.dataset import DatasetService
from symptom_diagnosis_explorer.services.ml_models.base import BaseModelService


class FrameworkNotRegisteredError(ValueError):
    """Raised when attempting to use an unregistered framework."""

    pass


class FrameworkRegistry:
    """Registry for model development services.

    This class provides a centralized registry for different ML framework
    implementations. Services register themselves using the decorator pattern,
    and the registry provides factory methods for creating service instances.

    Usage:
        # Register a service (typically in the service module itself)
        @FrameworkRegistry.register("dspy")
        class DSPyModelService(BaseModelService):
            ...

        # Create a service instance via the factory
        service = FrameworkRegistry.create_service(config, dataset_service)

    The registry uses the framework type from the config to determine which
    service class to instantiate.
    """

    _registry: dict[str, Type[BaseModelService]] = {}

    @classmethod
    def register(
        cls, framework_type: str
    ) -> Callable[[Type[BaseModelService]], Type[BaseModelService]]:
        """Decorator to register a service class for a framework.

        Args:
            framework_type: Framework identifier (e.g., "dspy", "langchain").

        Returns:
            Decorator function that registers the service class.

        Example:
            @FrameworkRegistry.register("dspy")
            class DSPyModelService(BaseModelService):
                ...
        """

        def decorator(
            service_class: Type[BaseModelService],
        ) -> Type[BaseModelService]:
            cls._registry[framework_type] = service_class
            return service_class

        return decorator

    @classmethod
    def create_service(
        cls,
        config: ClassificationConfig,
        dataset_service: DatasetService,
    ) -> BaseModelService:
        """Factory method to create a service instance based on config.

        This method inspects the config's framework_config to determine which
        framework to use, then instantiates the appropriate service class.

        Args:
            config: Classification configuration with framework-specific settings.
            dataset_service: Dataset service instance to inject.

        Returns:
            Instantiated service for the configured framework.

        Raises:
            FrameworkNotRegisteredError: If the framework type is not registered.
        """
        # Extract framework type from the discriminated union
        framework_type = config.framework_config.framework

        if framework_type not in cls._registry:
            available = ", ".join(sorted(cls._registry.keys()))
            raise FrameworkNotRegisteredError(
                f"Framework '{framework_type}' is not registered. "
                f"Available frameworks: {available}"
            )

        service_class = cls._registry[framework_type]
        service = service_class(config)

        # Inject the dataset service (allows for dependency injection in tests)
        service.dataset_service = dataset_service

        return service

    @classmethod
    def list_frameworks(cls) -> list[str]:
        """List all registered framework types.

        Returns:
            Sorted list of framework identifiers.
        """
        return sorted(cls._registry.keys())

    @classmethod
    def get_service_class(cls, framework_type: str) -> Type[BaseModelService]:
        """Get the service class for a framework type.

        Args:
            framework_type: Framework identifier.

        Returns:
            Service class for the framework.

        Raises:
            FrameworkNotRegisteredError: If the framework type is not registered.
        """
        if framework_type not in cls._registry:
            available = ", ".join(sorted(cls._registry.keys()))
            raise FrameworkNotRegisteredError(
                f"Framework '{framework_type}' is not registered. "
                f"Available frameworks: {available}"
            )

        return cls._registry[framework_type]
