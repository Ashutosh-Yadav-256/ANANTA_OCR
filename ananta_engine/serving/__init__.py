"""Serving subpackage for ANANTA OCR."""

from .client import AnantaClient, OCRClient, resolve_device
from .recognizer_http import HttpRecognizer, DEFAULT_MODEL, build_messages, encode_crop

__all__ = [
    "AnantaClient",
    "OCRClient",
    "resolve_device",
    "HttpRecognizer",
    "DEFAULT_MODEL",
    "build_messages",
    "encode_crop",
]
