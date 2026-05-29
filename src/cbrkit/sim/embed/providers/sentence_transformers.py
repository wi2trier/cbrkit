"""sentence-transformers dense embedding provider."""

from collections.abc import Sequence
from dataclasses import dataclass
from typing import Literal, cast, override

from sentence_transformers import SentenceTransformer

from ....typing import BatchConversionFunc, HasMetadata, JsonDict, NumpyArray


@dataclass(slots=True)
class sentence_transformers(BatchConversionFunc[str, NumpyArray], HasMetadata):
    """Semantic similarity using [sentence-transformers](https://www.sbert.net/)

    Args:
        model: Either the name of a [pretrained model](https://www.sbert.net/docs/pretrained_models.html)
            or a `SentenceTransformer` model instance.
    """

    model: SentenceTransformer
    batch_size: int
    show_progress_bar: bool | None
    precision: Literal["float32", "int8", "uint8", "binary", "ubinary"]
    truncate_dim: int | None
    normalize_embeddings: bool
    _metadata: JsonDict

    def __init__(
        self,
        model: str | SentenceTransformer,
        batch_size: int = 32,
        show_progress_bar: bool | None = None,
        precision: Literal["float32", "int8", "uint8", "binary", "ubinary"] = "float32",
        truncate_dim: int | None = None,
        normalize_embeddings: bool = False,
    ):
        self._metadata = {}
        self.batch_size = batch_size
        self.show_progress_bar = show_progress_bar
        self.precision = precision
        self.truncate_dim = truncate_dim
        self.normalize_embeddings = normalize_embeddings

        if isinstance(model, str):
            self.model = SentenceTransformer(model)
            self._metadata["model"] = model
        else:
            self.model = model
            self._metadata["model"] = (
                model.model_card_data.model_id
                or model.model_card_data.base_model
                or "custom"
            )

    @property
    @override
    def metadata(self) -> JsonDict:
        """Return metadata describing the sentence-transformers model."""
        return self._metadata

    @override
    def __call__(self, texts: Sequence[str]) -> Sequence[NumpyArray]:
        if not texts:
            return []

        vecs = self.model.encode(
            cast(list[str], texts),
            convert_to_numpy=True,
            batch_size=self.batch_size,
            show_progress_bar=self.show_progress_bar,
            precision=self.precision,
            truncate_dim=self.truncate_dim,
            normalize_embeddings=self.normalize_embeddings,
        )

        return list(vecs)


__all__ = ["sentence_transformers"]
