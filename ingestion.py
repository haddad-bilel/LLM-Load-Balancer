from __future__ import annotations

import os
from pathlib import Path
from typing import Iterable, List

from langchain_core.documents import Document
from langchain_community.document_loaders import TextLoader, PyPDFLoader, Docx2txtLoader

SUPPORTED_EXTENSIONS = {".txt", ".md", ".pdf", ".docx"}


def discover_files(input_path: str | Path) -> List[Path]:
    path = Path(input_path)
    if path.is_file():
        return [path]
    files: List[Path] = []
    for ext in SUPPORTED_EXTENSIONS:
        files.extend(path.rglob(f"*{ext}"))
    return files


def load_document(file_path: str | Path) -> List[Document]:
    path = Path(file_path)
    suffix = path.suffix.lower()
    metadata = {"source": str(path)}
    if suffix in {".txt", ".md"}:
        loader = TextLoader(str(path), encoding="utf-8")
    elif suffix == ".pdf":
        loader = PyPDFLoader(str(path))
    elif suffix == ".docx":
        loader = Docx2txtLoader(str(path))
    else:
        raise ValueError(f"Unsupported file type: {suffix}")
    docs = loader.load()
    for d in docs:
        d.metadata.update(metadata)
    return docs


def load_documents(input_path: str | Path) -> List[Document]:
    files = discover_files(input_path)
    all_docs: List[Document] = []
    for f in files:
        all_docs.extend(load_document(f))
    return all_docs


