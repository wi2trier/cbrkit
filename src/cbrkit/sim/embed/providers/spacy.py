"""SpaCy-based embedding provider."""

from collections.abc import Iterator, Sequence
from contextlib import AbstractContextManager
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any, override

import numpy as np
import spacy as spacylib
from spacy.cli.download import get_latest_version, get_model_filename
from spacy.language import Language

from ....constants import CACHE_DIR
from ....typing import BatchConversionFunc, HasMetadata, JsonDict, NumpyArray


def load_spacy(name: str | None, cache_dir: Path = CACHE_DIR) -> Language:
    """Load a spaCy model by name, downloading it if necessary."""
    import tarfile
    import urllib.request

    from rich.progress import Progress, TaskID

    @dataclass(slots=True)
    class ProgressHook(AbstractContextManager[Any]):
        """Progress reporting hook for URL downloads."""

        description: str
        progress: Progress = field(default_factory=Progress, init=False)
        task: TaskID | None = field(default=None, init=False)

        def __enter__(self):
            super().__enter__()
            self.progress.start()
            return self

        def __exit__(self, exc_type, exc_value, traceback):
            self.progress.stop()

        def __call__(self, block_num: int, block_size: int, total_size: int):
            if self.task is None:
                self.task = self.progress.add_task(self.description, total=total_size)

            downloaded = block_num * block_size

            if downloaded < total_size:
                self.progress.update(self.task, completed=downloaded)

            if self.progress.finished:
                self.task = None

    def tarfile_members(tf: tarfile.TarFile, prefix: str) -> Iterator[tarfile.TarInfo]:
        """Yield tar members with the given prefix stripped from their paths."""
        prefix_len = len(prefix)

        for member in tf.getmembers():
            if member.path.startswith(prefix):
                member.path = member.path[prefix_len:]

                yield member

    if not name:
        return spacylib.blank("en")

    version = get_latest_version(name)
    filename = get_model_filename(name, version, sdist=True)
    versioned_name = f"{name}-{version}"
    cache_file = cache_dir / "spacy" / versioned_name
    tmpfile = cache_file.with_suffix(".tar.gz")

    if not cache_file.exists():
        cache_file.parent.mkdir(parents=True, exist_ok=True)
        download_url = f"{spacylib.about.__download_url__}/{filename}"

        with ProgressHook(
            f"Downloading '{versioned_name}' to '{cache_file.parent}'..."
        ) as hook:
            urllib.request.urlretrieve(download_url, tmpfile, hook)

        with tarfile.open(tmpfile, mode="r:gz") as tf:
            member_prefix = f"{versioned_name}/{name}/{versioned_name}/"
            members = tarfile_members(tf, member_prefix)
            tf.extractall(path=cache_file, members=members)

        tmpfile.unlink()

    return spacylib.load(cache_file)


@dataclass(slots=True)
class spacy(BatchConversionFunc[str, NumpyArray], HasMetadata):
    """Semantic similarity using [spaCy](https://spacy.io/)

    Args:
        model: Either the name of a [spaCy model](https://spacy.io/usage/models)
            or a `spacy.Language` model instance.
    """

    model: Language

    def __init__(self, model: str | Language):
        if isinstance(model, str):
            self.model = load_spacy(model)
        else:
            self.model = model

    @property
    @override
    def metadata(self) -> JsonDict:
        """Return metadata describing the spaCy model."""
        return {
            "model": self.model.meta if isinstance(self.model, Language) else "custom"
        }

    @override
    def __call__(self, texts: Sequence[str]) -> Sequence[NumpyArray]:
        with self.model.select_pipes(enable=[]):
            docs_iterator = self.model.pipe(texts)

        return [np.asarray(doc.vector, dtype=np.float64) for doc in docs_iterator]


__all__ = ["spacy", "load_spacy"]
