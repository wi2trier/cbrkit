from collections.abc import Sequence
from dataclasses import dataclass
from typing import cast, override

from sentence_transformers import SentenceTransformer, util

from ...helpers import dispatch_batches, get_logger
from ...typing import Casebase, HasMetadata, JsonDict, RetrieverFunc

logger = get_logger(__name__)


@dataclass(slots=True, frozen=True)
class sentence_transformers[K](
    RetrieverFunc[K, str, float],
    HasMetadata,
):
    """Semantic similarity using sentence transformers

    Args:
        model: Name of the [sentence transformer model](https://www.sbert.net/docs/pretrained_models.html).
    """

    model: SentenceTransformer | str
    query_chunk_size: int = 100
    corpus_chunk_size: int = 500000
    device: str | None = None

    @property
    @override
    def metadata(self) -> JsonDict:
        """Return metadata describing the sentence transformer configuration."""
        return {
            "model": self.model if isinstance(self.model, str) else "custom",
            "query_chunk_size": self.query_chunk_size,
            "corpus_chunk_size": self.corpus_chunk_size,
            "device": self.device,
        }

    @override
    def __call__(
        self,
        batches: Sequence[tuple[Casebase[K, str], str]],
    ) -> Sequence[tuple[Casebase[K, str], dict[K, float]]]:
        if not batches:
            return []

        if isinstance(self.model, str):
            model = SentenceTransformer(self.model, device=self.device)
        else:
            model = self.model

        model.to(self.device)

        sim_maps = dispatch_batches(
            batches,
            lambda queries, casebase: self.__call_queries__(queries, casebase, model),
        )

        return [
            (casebase, sim_map)
            for (casebase, _), sim_map in zip(batches, sim_maps, strict=True)
        ]

    def __call_queries__(
        self,
        queries: Sequence[str],
        casebase: Casebase[K, str],
        model: SentenceTransformer,
    ) -> Sequence[dict[K, float]]:
        case_texts = list(casebase.values())
        query_texts = cast(list[str], queries)

        case_embeddings = util.normalize_embeddings(
            model.encode(case_texts, convert_to_tensor=True).to(self.device)
        )
        query_embeddings = util.normalize_embeddings(
            model.encode(query_texts, convert_to_tensor=True).to(self.device)
        )

        response = util.semantic_search(
            query_embeddings,
            case_embeddings,
            top_k=len(casebase),
            query_chunk_size=self.query_chunk_size,
            corpus_chunk_size=self.corpus_chunk_size,
            score_function=util.dot_score,
        )

        key_index = dict(enumerate(casebase))

        return [
            {
                key_index[cast(int, res["corpus_id"])]: float(res["score"])
                for res in query_response
            }
            for query_response in response
        ]


__all__ = ["sentence_transformers"]
