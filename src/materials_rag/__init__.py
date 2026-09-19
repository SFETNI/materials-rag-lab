"""Public interface for Materials RAG Lab."""

from importlib.metadata import PackageNotFoundError, version

try:
    __version__ = version("materials-rag")
except PackageNotFoundError:  # Source-tree use before installation.
    __version__ = "0.1.0"

from materials_rag.api import MaterialsRAG

__all__ = ["MaterialsRAG", "__version__"]
