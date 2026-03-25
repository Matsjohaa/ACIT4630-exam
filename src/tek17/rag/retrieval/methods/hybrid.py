from __future__ import annotations

from typing import Any

import chromadb

from tek17.rag.retrieval.methods.dense import retrieve_dense
from tek17.rag.retrieval.methods.sparse import retrieve_sparce


def _chunk_key(doc: str, meta: dict[str, Any]) -> str:
	section_id = str(meta.get("section_id", ""))
	text_type = str(meta.get("text_type", ""))
	para_start = str(meta.get("para_start", ""))
	para_end = str(meta.get("para_end", ""))
	title = str(meta.get("title", ""))
	base = "|".join([section_id, title, text_type, para_start, para_end])
	if base.strip("|"):
		return base
	# fallback
	return str(hash(doc))


def retrieve_hybrid(
	collection: chromadb.Collection,
	query_text: str,
	query_embedding: list[float],
	top_k: int,
	*,
	alpha: float = 0.5,
	dense_candidates: int | None = None,
	sparce_candidates: int | None = None,
	chunks_path=None,
) -> tuple[list[str], list[dict[str, Any]], list[float]]:
	"""Hybrid retrieval = combine dense + sparse scores.

	alpha in [0,1] weights dense vs sparse.
	Returns distances in [0,1] where lower is better.
	"""

	if chunks_path is None:
		raise ValueError("chunks_path is required for hybrid retrieval")

	cand_d = dense_candidates or max(top_k * 3, top_k)
	cand_s = sparce_candidates or max(top_k * 3, top_k)

	d_docs, d_metas, d_dists = retrieve_dense(collection, query_embedding, cand_d)
	s_docs, s_metas, s_dists = retrieve_sparce(query_text, cand_s, chunks_path)

	# Convert distances->scores (higher better)
	dense_scores: dict[str, float] = {}
	dense_payload: dict[str, tuple[str, dict[str, Any], float]] = {}
	for doc, meta, dist in zip(d_docs, d_metas, d_dists):
		key = _chunk_key(doc, meta)
		score = 1.0 / (1.0 + float(dist))  # monotonic
		if key not in dense_scores or score > dense_scores[key]:
			dense_scores[key] = score
			dense_payload[key] = (doc, meta, float(dist))

	sparse_scores: dict[str, float] = {}
	sparse_payload: dict[str, tuple[str, dict[str, Any], float]] = {}
	for doc, meta, dist in zip(s_docs, s_metas, s_dists):
		key = _chunk_key(doc, meta)
		score = 1.0 - float(dist)  # because dist in [0,1]
		if key not in sparse_scores or score > sparse_scores[key]:
			sparse_scores[key] = score
			sparse_payload[key] = (doc, meta, float(dist))

	keys = set(dense_scores) | set(sparse_scores)
	if not keys:
		return [], [], []

	max_d = max(dense_scores.values(), default=0.0) or 1.0
	max_s = max(sparse_scores.values(), default=0.0) or 1.0

	combined: list[tuple[str, float]] = []
	for k in keys:
		d = dense_scores.get(k, 0.0) / max_d
		s = sparse_scores.get(k, 0.0) / max_s
		c = (alpha * d) + ((1.0 - alpha) * s)
		combined.append((k, c))

	combined.sort(key=lambda x: x[1], reverse=True)
	combined = combined[:top_k]

	out_docs: list[str] = []
	out_metas: list[dict[str, Any]] = []
	out_dists: list[float] = []

	for k, c in combined:
		# prefer dense metadata/doc if available, otherwise sparse
		if k in dense_payload:
			doc, meta, _ = dense_payload[k]
		else:
			doc, meta, _ = sparse_payload[k]
		out_docs.append(doc)
		out_metas.append(meta)
		out_dists.append(1.0 - c)

	return out_docs, out_metas, out_dists

