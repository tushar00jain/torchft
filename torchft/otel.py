import json
import logging
import os
import time
from typing import List, Sequence

from opentelemetry._logs import set_logger_provider

from opentelemetry.exporter.otlp.proto.http._log_exporter import OTLPLogExporter
from opentelemetry.sdk._logs import LoggerProvider, LoggingHandler
from opentelemetry.sdk._logs._internal import LogData
from opentelemetry.sdk._logs.export import (
    BatchLogRecordProcessor,
    ConsoleLogExporter,
    LogExporter,
)
from opentelemetry.sdk.resources import Resource

_LOGGER_PROVIDER: LoggerProvider | None = None
# Path to the file containing OTEL resource attributes
TORCHFT_OTEL_RESOURCE_ATTRIBUTES_JSON = "TORCHFT_OTEL_RESOURCE_ATTRIBUTES_JSON"


class TeeLogExporter(LogExporter):
    """Exporter that writes to multiple exporters."""

    def __init__(
        self,
        exporters: List[LogExporter],
    ) -> None:
        self._exporters = exporters

    def export(self, batch: Sequence[LogData]) -> None:
        for e in self._exporters:
            e.export(batch)

    def shutdown(self) -> None:
        for e in self._exporters:
            e.shutdown()


def setup_logger() -> None:
    torchft_otel_resource_attributes_json = os.environ.get(
        TORCHFT_OTEL_RESOURCE_ATTRIBUTES_JSON
    )

    if torchft_otel_resource_attributes_json is not None:
        with open(torchft_otel_resource_attributes_json) as f:
            attributes = json.loads(f.read())
            resource = Resource.create(attributes=attributes)
    else:
        resource = Resource.create()

    logger_provider = LoggerProvider(resource=resource)
    set_logger_provider(logger_provider)

    exporter = TeeLogExporter(
        exporters=[
            ConsoleLogExporter(),
            OTLPLogExporter(
                timeout=5,
            ),
        ],
    )
    logger_provider.add_log_record_processor(BatchLogRecordProcessor(exporter))
    handler = LoggingHandler(level=logging.NOTSET, logger_provider=logger_provider)

    logging.getLogger().setLevel(logging.NOTSET)

    # Attach OTLP handler to root logger
    logging.getLogger().addHandler(handler)

    global _LOGGER_PROVIDER
    _LOGGER_PROVIDER = logger_provider


def shutdown() -> None:
    assert _LOGGER_PROVIDER is not None
    _LOGGER_PROVIDER.shutdown()


# Example usage of the logger
def main() -> None:
    setup_logger()

    while True:
        time.sleep(1)
        logging.debug(
            "Quick zephyrs blow, vexing daft Jim.",
            extra={
                "test_attr": "value1",
            },
        )
        logging.info("Jackdaws love my big sphinx of quartz.")

        # Create different namespaced logger
        logger = logging.getLogger("myapp.area1")

        logger.debug(
            "Quick zephyrs blow, vexing daft Jim.",
            extra={
                "test_attr": "value2",
            },
        )
        logger.info("How quickly daft jumping zebras vex.")

        print("Example done; exiting...")


if __name__ == "__main__":
    main()
