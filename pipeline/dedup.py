"""Dedup bằng embedding (bge-m3) — dùng chung cho expand (C1) và filter (D2).

Ý tưởng: nhúng mọi text thành vector, duyệt lần lượt, item nào "gần trùng"
(cosine similarity vượt ngưỡng) với một item ĐÃ GIỮ thì bỏ.
"""
from functools import lru_cache

import numpy as np

from config import EMBED_MODEL


@lru_cache(maxsize=1)
def _embedder():
    """Nạp model embedding một lần duy nhất (lười — chỉ tải khi cần)."""
    from sentence_transformers import SentenceTransformer
    print(f"  (nạp embedding model {EMBED_MODEL}, lần đầu sẽ tải về...)")
    return SentenceTransformer(EMBED_MODEL)


def dedup_texts(items, key, threshold):
    """Giữ lại các item mà text chưa gần trùng với item nào đã giữ.

    items     : list bất kỳ
    key       : hàm lấy chuỗi text từ một item
    threshold : cosine sim >= ngưỡng này -> coi là trùng, loại
    """
    if len(items) <= 1:
        return list(items)

    texts = [key(it) for it in items]
    emb = _embedder().encode(texts, normalize_embeddings=True,
                             show_progress_bar=False)
    emb = np.asarray(emb, dtype=np.float32)

    kept = []          # chỉ số các item được giữ
    for i in range(len(items)):
        if kept:
            # vì vector đã chuẩn hoá -> tích vô hướng chính là cosine sim
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
        vecs = _embedder().encode(texts, normalize_embeddings=True,
                                  show_progress_bar=False)
        vecs = np.asarray(vecs, dtype=np.float32)
        self._kept = vecs if self._kept is None else np.vstack([self._kept, vecs])

    def filter_new(self, texts):
        """Trả các text KHÔNG gần trùng (với text đã giữ trước đó và với nhau).

        Các text được giữ sẽ được nhớ lại cho các lần gọi sau.
        """
        texts = [t for t in texts if t]
        if not texts:
            return []
        vecs = _embedder().encode(texts, normalize_embeddings=True,
                                  show_progress_bar=False)
        vecs = np.asarray(vecs, dtype=np.float32)

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
