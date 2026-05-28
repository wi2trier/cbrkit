"""sentence-transformers sparse embedding provider."""

from collections.abc import Sequence
from dataclasses import dataclass
from typing import Any, cast, override

from sentence_transformers.sparse_encoder import SparseEncoder

from ....typing import BatchConversionFunc, HasMetadata, JsonDict, SparseVector


@dataclass(slots=True)
class sparse_encoder(BatchConversionFunc[str, SparseVector], HasMetadata):
    """Sparse embeddings using `sentence-transformers <https://www.sbert.net/>`_ SparseEncoder.

    Wraps any `SparseEncoder` model, including SPLADE variants and other
    sparse embedding models.  Produces sparse vectors where each dimension
    corresponds to a vocabulary token and the value represents the token's
    importance.

    Args:
        model: Either the name of a sparse model (e.g.,
            `"naver/splade-cocondenser-ensembledistil"`) or a
            `SparseEncoder` instance.
        batch_size: Batch size for encoding.
        show_progress_bar: Whether to show a progress bar.
    """

    model: SparseEncoder
    batch_size: int
    show_progress_bar: bool | None
    _metadata: JsonDict

    def __init__(
        self,
        model: str | SparseEncoder,
        batch_size: int = 32,
        show_progress_bar: bool | None = None,
    ):
        self._metadata = {}
        self.batch_size = batch_size
        self.show_progress_bar = show_progress_bar

        if isinstance(model, str):
            self.model = SparseEncoder(model)
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
        """Return metadata describing the sparse encoder model."""
        return self._metadata

    @override
    def __call__(self, texts: Sequence[str]) -> Sequence[SparseVector]:
        if not texts:
            return []

        embeddings = cast(
            Any,
            self.model.encode(
                cast(list[str], texts),
                batch_size=self.batch_size,
                show_progress_bar=self.show_progress_bar,
                convert_to_sparse_tensor=True,
            ),
        )

        # Convert 2D sparse COO tensor to list of {token_id: weight}
        coalesced = embeddings.coalesce()
        indices = coalesced.indices()  # [2, nnz]
        values = coalesced.values()  # [nnz]

        result: list[SparseVector] = [{} for _ in range(len(texts))]

        for idx in range(indices.shape[1]):
            row = int(indices[0, idx])
            col = int(indices[1, idx])
            val = float(values[idx])
            if val != 0.0:
                result[row][col] = val

        return result


__all__ = ["sparse_encoder"]
