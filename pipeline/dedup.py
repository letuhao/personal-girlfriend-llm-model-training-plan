"""Dedup bằng embedding — dùng chung cho expand (C1) và filter (D2).

Gọi LM Studio embeddings API (text-embedding-bge-m3 đã loaded sẵn) thay vì
load local SentenceTransformer — tránh tranh VRAM với teacher model 35B.

Ý tưởng: nhúng mọi text thành vector, duyệt lần lượt, item nào "gần trùng"
(cosine similarity vượt ngưỡng) với một item ĐÃ GIỮ thì bỏ.
"""
from functools import lru_cache

import numpy as np
from openai import OpenAI

from config import EMBED_MODEL, LMSTUDIO_BASE_URL, LMSTUDIO_API_KEY


@lru_cache(maxsize=1)
def _client() -> OpenAI:
    return OpenAI(base_url=LMSTUDIO_BASE_URL, api_key=LMSTUDIO_API_KEY)


def _embed(texts: list[str]) -> np.ndarray:
    """Gọi LM Studio embeddings API, trả ma trận đã normalize (n, dim)."""
    resp = _client().embeddings.create(model=EMBED_MODEL, input=texts)
    # Giữ đúng thứ tự theo index trả về
    vecs = np.array(
        [d.embedding for d in sorted(resp.data, key=lambda x: x.index)],
        dtype=np.float32,
    )
    norms = np.linalg.norm(vecs, axis=1, keepdims=True)
    return vecs / np.where(norms == 0, 1.0, norms)


def dedup_texts(items, key, threshold):
    """Giữ lại các item mà text chưa gần trùng với item nào đã giữ.

    items     : list bất kỳ
    key       : hàm lấy chuỗi text từ một item
    threshold : cosine sim >= ngưỡng này -> coi là trùng, loại
    """
    if len(items) <= 1:
        return list(items)

    texts = [key(it) for it in items]
    emb = _embed(texts)

    kept = []
    for i in range(len(items)):
        if kept:
            sims = emb[kept] @ emb[i]
            if float(sims.max()) >= threshold:
                continue
        kept.append(i)
    return [items[i] for i in kept]


class Deduper:
    """Dedup ngữ nghĩa TĂNG DẦN — nhận text theo từng đợt, nhớ embedding đã giữ.

    Dùng cho expand: mỗi batch mới chỉ embed phần mới rồi so với phần đã giữ,
    không phải embed lại toàn bộ. Hợp với việc ghi file tăng dần (resume).
    """

    def __init__(self, threshold):
        self.threshold = threshold
        self._kept = None        # ndarray (n, dim) embedding các text đã giữ

    def prime(self, texts):
        """Nạp sẵn text đã có (vd khi resume) vào bộ nhớ — không lọc, không trả."""
        texts = [t for t in texts if t]
        if not texts:
            return
        # Gọi theo batch 512 để tránh request quá lớn
        vecs = np.vstack([_embed(texts[i:i+512]) for i in range(0, len(texts), 512)])
        self._kept = vecs if self._kept is None else np.vstack([self._kept, vecs])

    def filter_new(self, texts):
        """Trả các text KHÔNG gần trùng (với text đã giữ trước đó và với nhau).

        Các text được giữ sẽ được nhớ lại cho các lần gọi sau.
        """
        texts = [t for t in texts if t]
        if not texts:
            return []
        vecs = _embed(texts)

        kept_txt, kept_vec = [], []
        for t, v in zip(texts, vecs):
            ref = []
            if self._kept is not None:
                ref.append(self._kept)
            if kept_vec:
                ref.append(np.stack(kept_vec))
            if ref and float((np.vstack(ref) @ v).max()) >= self.threshold:
                continue
            kept_txt.append(t)
            kept_vec.append(v)

        if kept_vec:
            block = np.stack(kept_vec)
            self._kept = block if self._kept is None else np.vstack([self._kept, block])
        return kept_txt
