"""Индекс резюме: TF-IDF из курсача кладётся в Qdrant, поиск идёт через query_points."""

from __future__ import annotations

import re
import sys
import threading
from pathlib import Path

import numpy as np
from qdrant_client import QdrantClient, models

ROOT = Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

import kurs  # noqa: E402

COLLECTION = "resumes"


def verdict(score: float) -> str:
    """Те же пороги, что у PairScore в kurs.py."""
    if score >= 0.62:
        return "рекомендовать"
    if score >= 0.40:
        return "рассмотреть"
    return "слабое соответствие"


class ResumeIndex:
    """Векторы резюме лежат в локальном Qdrant. Вакансия — это вектор запроса."""

    def __init__(self) -> None:
        self.client = QdrantClient(":memory:")
        self._lock = threading.Lock()
        self.resumes: list[kurs.Resume] = []
        self.resume_matrix = np.zeros((0, 0))
        self.feature_names: np.ndarray = np.array([])
        self.vacancy_vectors: dict[str, list[float]] = {}
        self.vacancy_skills: dict[str, list[str]] = {}
        self.reload()

    def reload(self) -> None:
        with self._lock:
            self._rebuild()

    def _rebuild(self) -> None:
        self.resumes = list(kurs.RESUMES) + list(kurs.load_resumes(kurs.RESUMES_DIR))
        resume_texts = [kurs.normalize(item.text) for item in self.resumes]
        vacancy_texts = [kurs.normalize(item.text) for item in kurs.VACANCIES]
        vectorizer, matrix = kurs.build_vectorizer(resume_texts + vacancy_texts)
        resume_count = len(self.resumes)
        self.resume_matrix = matrix[:resume_count].toarray()
        vacancy_matrix = matrix[resume_count:].toarray()
        self.feature_names = vectorizer.get_feature_names_out()
        self.vacancy_vectors = {
            vacancy.code: vacancy_matrix[index].astype(float).tolist()
            for index, vacancy in enumerate(kurs.VACANCIES)
        }
        self.vacancy_skills = {
            vacancy.code: kurs.extract_skills(vacancy.text) for vacancy in kurs.VACANCIES
        }
        self._recreate_collection(int(self.resume_matrix.shape[1]))
        if resume_count == 0:
            return
        points = [
            models.PointStruct(
                id=index,
                vector=self.resume_matrix[index].astype(float).tolist(),
                payload={
                    "row": index,
                    "code": resume.code,
                    "name": resume.name,
                    "text": resume.text.strip(),
                    "skills": kurs.extract_skills(resume.text),
                },
            )
            for index, resume in enumerate(self.resumes)
        ]
        self.client.upsert(collection_name=COLLECTION, points=points)

    def _recreate_collection(self, size: int) -> None:
        if size <= 0:
            size = 1
        if self.client.collection_exists(COLLECTION):
            self.client.delete_collection(COLLECTION)
        self.client.create_collection(
            collection_name=COLLECTION,
            vectors_config=models.VectorParams(size=size, distance=models.Distance.COSINE),
        )

    def add_resume(self, name: str, text: str) -> kurs.Resume:
        name = name.strip()
        text = text.strip()
        if not name or not text:
            raise ValueError("Нужны имя и текст резюме")
        code = self._new_code(name)
        kurs.RESUMES_DIR.mkdir(parents=True, exist_ok=True)
        path = kurs.RESUMES_DIR / f"{code}.txt"
        path.write_text(f"Имя: {name}\n\n{text}\n", encoding="utf-8")
        self.reload()
        return next(item for item in self.resumes if item.code == code)

    def _new_code(self, name: str) -> str:
        base = re.sub(r"\s+", "_", name.strip().lower())
        base = re.sub(r"[^\w\-]+", "", base, flags=re.UNICODE).strip("_") or "kandidat"
        taken = {item.code for item in self.resumes}
        taken.update(path.stem for path in kurs.RESUMES_DIR.glob("*.txt"))
        candidate = base
        number = 2
        while candidate in taken:
            candidate = f"{base}_{number}"
            number += 1
        return candidate

    def match(self, vacancy_code: str) -> list[dict]:
        with self._lock:
            if vacancy_code not in self.vacancy_vectors:
                raise KeyError(vacancy_code)
            if not self.resumes:
                return []
            found = self.client.query_points(
                collection_name=COLLECTION,
                query=self.vacancy_vectors[vacancy_code],
                limit=len(self.resumes),
                with_payload=True,
            )
            required = self.vacancy_skills[vacancy_code]
            query_vector = np.array(self.vacancy_vectors[vacancy_code])
            rows: list[dict] = []
            for cosine_rank, point in enumerate(found.points, start=1):
                payload = point.payload or {}
                have = set(payload.get("skills") or [])
                matched = [skill for skill in required if skill in have]
                missing = [skill for skill in required if skill not in have]
                coverage = len(matched) / len(required) if required else 0.0
                cosine = float(point.score)
                score = kurs.TEXT_WEIGHT * cosine + kurs.SKILL_WEIGHT * coverage
                rows.append(
                    {
                        "code": payload.get("code", ""),
                        "name": payload.get("name", ""),
                        "cosine": round(cosine, 3),
                        "cosine_rank": cosine_rank,
                        "coverage": round(coverage, 3),
                        "score": round(score, 3),
                        "verdict": verdict(score),
                        "matched": matched,
                        "missing": missing,
                        "terms": self._terms(int(payload.get("row", 0)), query_vector),
                        "excerpt": _excerpt(str(payload.get("text", ""))),
                    }
                )
        rows.sort(key=lambda row: row["score"], reverse=True)
        return rows

    def _terms(self, row: int, query_vector: np.ndarray) -> list[str]:
        if row < 0 or row >= len(self.resume_matrix):
            return []
        product = self.resume_matrix[row] * query_vector
        order = np.argsort(product)[::-1][:6]
        terms: list[str] = []
        for index in order:
            weight = float(product[index])
            if weight <= 0:
                break
            terms.append(f"{self.feature_names[index]} ({weight:.3f})")
        return terms


def _excerpt(text: str, limit: int = 220) -> str:
    compact = re.sub(r"\s+", " ", text).strip()
    if len(compact) <= limit:
        return compact
    return compact[: limit - 1].rstrip() + "…"
