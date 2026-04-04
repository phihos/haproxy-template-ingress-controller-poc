"""
Distributed tracing functionality for HAProxy Template IC.

This module provides OpenTelemetry-based distributed tracing to enable
end-to-end observability across the entire template rendering and deployment
pipeline. It includes automatic instrumentation for async operations, HTTP
requests, and custom business logic spans.
"""

import functools
import os
from contextlib import contextmanager
from dataclasses import dataclass
from typing import Any, Awaitable, Callable, Iterator, TypeVar, cast

import structlog
from opentelemetry import trace
from opentelemetry.exporter.otlp.proto.grpc.trace_exporter import OTLPSpanExporter
from opentelemetry.instrumentation.asyncio import AsyncioInstrumentor
from opentelemetry.instrumentation.httpx import HTTPXClientInstrumentor
from opentelemetry.sdk.resources import Resource
from opentelemetry.sdk.trace import TracerProvider
from opentelemetry.sdk.trace.export import BatchSpanProcessor, ConsoleSpanExporter
from opentelemetry.trace import Status, StatusCode

from haproxy_template_ic.core.error_handling import handle_exceptions

SERVICE_NAME = "service.name"
SERVICE_VERSION = "service.version"
SERVICE_INSTANCE_ID = "service.instance.id"


logger = structlog.get_logger(__name__)

F = TypeVar("F", bound=Callable[..., Any])
AsyncF = TypeVar("AsyncF", bound=Callable[..., Awaitable[Any]])


@dataclass
class TracingConfig:
    """Configuration for distributed tracing."""

    enabled: bool = False
    service_name: str = "haproxy-template-ic"
    service_version: str = "1.0.0"
    jaeger_endpoint: str | None = None
    sample_rate: float = 1.0
    console_export: bool = False

    def override_with_app_config(self, app_tracing_config: Any) -> "TracingConfig":
        """Create new TracingConfig with app-specific overrides.

        Args:
            app_tracing_config: Application tracing configuration object

        Returns:
            New TracingConfig instance with app-specific values applied
        """
        return TracingConfig(
            enabled=app_tracing_config.enabled,
            service_name=app_tracing_config.service_name or self.service_name,
            service_version=app_tracing_config.service_version or self.service_version,
            jaeger_endpoint=app_tracing_config.jaeger_endpoint or self.jaeger_endpoint,
            sample_rate=app_tracing_config.sample_rate,
            console_export=app_tracing_config.console_export or self.console_export,
        )


class TracingManager:
    """Manages OpenTelemetry tracing configuration and lifecycle."""

    def __init__(self, config: TracingConfig):
        self.config = config
        self.tracer_provider: TracerProvider | None = None
        self.tracer: trace.Tracer | None = None
        self._instrumented = False

    def initialize(self) -> None:
        """Initialize distributed tracing with configured exporters."""
        if not self.config.enabled:
            logger.info("Distributed tracing is disabled")
            return

        logger.info(
            "Initializing distributed tracing",
            service_name=self.config.service_name,
            jaeger_endpoint=self.config.jaeger_endpoint,
        )

        resource = Resource(
            {
                SERVICE_NAME: self.config.service_name,
                SERVICE_VERSION: self.config.service_version,
                SERVICE_INSTANCE_ID: os.getenv("HOSTNAME", "unknown"),
            }
        )

        # Configure tracer provider
        self.tracer_provider = TracerProvider(resource=resource)
        trace.set_tracer_provider(self.tracer_provider)

        # Add span processors/exporters
        self._configure_exporters()

        self.tracer = trace.get_tracer(__name__)

        # Auto-instrument libraries
        self._setup_auto_instrumentation()

        logger.info("Distributed tracing initialized successfully")

    def _configure_exporters(self) -> None:
        """Configure span exporters based on configuration."""
        if not self.tracer_provider:
            return

        # Console exporter for development
        if self.config.console_export:
            console_exporter = ConsoleSpanExporter()
            console_processor = BatchSpanProcessor(console_exporter)
            self.tracer_provider.add_span_processor(console_processor)
            logger.debug("Added console span exporter")

        # OTLP exporter for production (e.g., Jaeger with OTLP support)
        if self.config.jaeger_endpoint:
            self._setup_otlp_exporter()

    @handle_exceptions(logger=logger, context="OTLP exporter configuration")
    def _setup_otlp_exporter(self) -> None:
        """Set up OTLP exporter with error handling."""
        if not self.config.jaeger_endpoint:
            return

        endpoint = self.config.jaeger_endpoint
        otlp_exporter = OTLPSpanExporter(endpoint=endpoint, insecure=True)
        otlp_processor = BatchSpanProcessor(otlp_exporter)
        if self.tracer_provider:
            self.tracer_provider.add_span_processor(otlp_processor)
        logger.info("Added OTLP span exporter", endpoint=endpoint)

    @handle_exceptions(logger=logger, context="auto-instrumentation setup")
    def _setup_auto_instrumentation(self) -> None:
        """Set up automatic instrumentation for common libraries."""
        if self._instrumented:
            return

        # Instrument HTTPX for Dataplane API calls
        HTTPXClientInstrumentor().instrument()
        logger.debug("HTTPX instrumentation enabled")

        # Instrument asyncio for async operation tracing
        AsyncioInstrumentor().instrument()
        logger.debug("Asyncio instrumentation enabled")

        self._instrumented = True

    @handle_exceptions(logger=logger, context="tracing shutdown")
    def shutdown(self) -> None:
        """Shutdown tracing and flush any pending spans."""
        if self.tracer_provider:
            self.tracer_provider.shutdown()
            logger.info("Distributed tracing shutdown complete")


def get_tracer(tracing_manager: TracingManager | None = None) -> trace.Tracer | None:
    """Get the configured tracer instance."""
    if tracing_manager and tracing_manager.tracer:
        return tracing_manager.tracer
    return None


@contextmanager
def trace_operation(
    operation_name: str,
    attributes: dict[str, Any] | None = None,
    set_status_on_exception: bool = True,
    tracing_manager: TracingManager | None = None,
) -> Iterator[trace.Span | None]:
    """Context manager for tracing operations with error handling."""
    tracer = get_tracer(tracing_manager)
    if not tracer:
        yield None
        return

    with tracer.start_as_current_span(operation_name) as span:
        try:
            # Add custom attributes
            if attributes:
                for key, value in attributes.items():
                    span.set_attribute(str(key), str(value))

            yield span

        except Exception as e:
            if set_status_on_exception:
                span.set_status(Status(StatusCode.ERROR, str(e)))
                span.record_exception(e)
            raise


def trace_async_function(
    span_name: str | None = None,
    attributes: dict[str, Any] | None = None,
    tracing_manager: TracingManager | None = None,
) -> Callable[[AsyncF], AsyncF]:
    """Decorator for tracing async functions."""

    def decorator(func: AsyncF) -> AsyncF:
        operation_name = (
            span_name or f"{func.__module__}.{getattr(func, '__name__', 'unknown')}"
        )

        @functools.wraps(func)
        async def wrapper(*args: Any, **kwargs: Any) -> Any:
            with trace_operation(
                operation_name, attributes, tracing_manager=tracing_manager
            ) as span:
                if span:
                    # Add function metadata
                    span.set_attribute(
                        "function.name", getattr(func, "__name__", "unknown")
                    )
                    span.set_attribute("function.module", func.__module__)

                return await func(*args, **kwargs)

        return cast(AsyncF, wrapper)

    return decorator


def trace_function(
    span_name: str | None = None,
    attributes: dict[str, Any] | None = None,
    tracing_manager: TracingManager | None = None,
) -> Callable[[F], F]:
    """Decorator for tracing synchronous functions."""

    def decorator(func: F) -> F:
        operation_name = (
            span_name or f"{func.__module__}.{getattr(func, '__name__', 'unknown')}"
        )

        @functools.wraps(func)
        def wrapper(*args: Any, **kwargs: Any) -> Any:
            with trace_operation(
                operation_name, attributes, tracing_manager=tracing_manager
            ) as span:
                if span:
                    # Add function metadata
                    span.set_attribute(
                        "function.name", getattr(func, "__name__", "unknown")
                    )
                    span.set_attribute("function.module", func.__module__)

                return func(*args, **kwargs)

        return cast(F, wrapper)

    return decorator


def add_span_attributes(**attributes: Any) -> None:
    """Add attributes to the current active span."""
    current_span = trace.get_current_span()
    if current_span and current_span.is_recording():
        for key, value in attributes.items():
            current_span.set_attribute(str(key), str(value))


def record_span_event(name: str, attributes: dict[str, Any] | None = None) -> None:
    """Record an event on the current active span."""
    current_span = trace.get_current_span()
    if current_span and current_span.is_recording():
        current_span.add_event(name, attributes or {})


def set_span_error(error: Exception, description: str | None = None) -> None:
    """Mark the current span as having an error."""
    current_span = trace.get_current_span()
    if current_span and current_span.is_recording():
        current_span.set_status(Status(StatusCode.ERROR, description or str(error)))
        current_span.record_exception(error)


# Convenience context managers for common operations
@contextmanager
def trace_template_render(
    template_type: str,
    template_path: str = "",
    tracing_manager: TracingManager | None = None,
) -> Iterator[None]:
    """Context manager for tracing template rendering operations."""
    attributes = {
        "template.type": template_type,
        "template.path": template_path,
        "operation.category": "template_rendering",
    }

    with trace_operation(
        f"render_{template_type}_template", attributes, tracing_manager=tracing_manager
    ):
        yield


@contextmanager
def trace_dataplane_operation(
    operation: str,
    instance_url: str = "",
    tracing_manager: TracingManager | None = None,
) -> Iterator[None]:
    """Context manager for tracing Dataplane API operations."""
    attributes = {
        "dataplane.operation": operation,
        "dataplane.instance_url": instance_url,
        "operation.category": "dataplane_api",
    }

    with trace_operation(
        f"dataplane_{operation}", attributes, tracing_manager=tracing_manager
    ):
        yield


@contextmanager
def trace_kubernetes_operation(
    resource_type: str,
    namespace: str = "",
    name: str = "",
    tracing_manager: TracingManager | None = None,
) -> Iterator[None]:
    """Context manager for tracing Kubernetes operations."""
    attributes = {
        "k8s.resource.type": resource_type,
        "k8s.resource.namespace": namespace,
        "k8s.resource.name": name,
        "operation.category": "kubernetes",
    }

    with trace_operation(
        f"k8s_{resource_type}_operation", attributes, tracing_manager=tracing_manager
    ):
        yield


def initialize_tracing(config: TracingConfig) -> TracingManager:
    """Initialize tracing with the given configuration.

    Args:
        config: TracingConfig instance

    Returns:
        TracingManager instance (initialized if enabled)
    """
    manager = TracingManager(config)
    manager.initialize()
    return manager


def create_tracing_config_from_env() -> TracingConfig:
    """Create tracing configuration from environment variables."""
    return TracingConfig(
        enabled=os.getenv("TRACING_ENABLED", "false").lower() == "true",
        service_name=os.getenv("TRACING_SERVICE_NAME", "haproxy-template-ic"),
        service_version=os.getenv("TRACING_SERVICE_VERSION", "1.0.0"),
        jaeger_endpoint=os.getenv("JAEGER_ENDPOINT"),
        sample_rate=float(os.getenv("TRACING_SAMPLE_RATE", "1.0")),
        console_export=os.getenv("TRACING_CONSOLE_EXPORT", "false").lower() == "true",
    )
