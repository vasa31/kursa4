"""Бэкенд сопоставления резюме с вакансиями 1С."""

from __future__ import annotations

import textwrap

from fastapi import FastAPI, HTTPException
from pydantic import BaseModel, Field

from backend.index import ResumeIndex
import kurs

app = FastAPI(
    title="Сопоставление резюме разработчиков 1С",
    version="1.0.0",
    description=(
        "Тексты переводятся в TF-IDF, векторы резюме лежат в Qdrant. "
        "Близость к вакансии считается методом query_points. "
        "Итог = 0.6 * косинус Qdrant + 0.4 * покрытие компетенций."
    ),
)
index = ResumeIndex()


class ResumeIn(BaseModel):
    name: str = Field(min_length=1, max_length=120)
    text: str = Field(min_length=1)


class MatchIn(BaseModel):
    vacancy_code: str = Field(min_length=1)


def _vacancy_payload(vacancy: kurs.Vacancy) -> dict:
    return {
        "code": vacancy.code,
        "title": vacancy.title,
        "text": textwrap.dedent(vacancy.text).strip(),
        "skills": kurs.extract_skills(vacancy.text),
    }


def _match_or_404(vacancy_code: str) -> dict:
    vacancy = next((item for item in kurs.VACANCIES if item.code == vacancy_code), None)
    if vacancy is None:
        raise HTTPException(status_code=404, detail="Вакансия не найдена")
    return {
        "vacancy": _vacancy_payload(vacancy),
        "formula": "0.6 * косинус query_points + 0.4 * покрытие компетенций",
        "thresholds": {
            "рекомендовать": 0.62,
            "рассмотреть": 0.40,
        },
        "results": index.match(vacancy_code),
    }


@app.get("/health")
def health() -> dict:
    return {"ok": True, "resumes": len(index.resumes), "engine": "qdrant.query_points"}


@app.get("/vacancies")
def vacancies() -> list[dict]:
    return [_vacancy_payload(item) for item in kurs.VACANCIES]


@app.get("/resumes")
def resumes() -> list[dict]:
    return [{"code": item.code, "name": item.name} for item in index.resumes]


@app.post("/resumes", status_code=201)
def add_resume(body: ResumeIn) -> dict:
    try:
        created = index.add_resume(body.name, body.text)
    except ValueError as error:
        raise HTTPException(status_code=422, detail=str(error)) from error
    return {"code": created.code, "name": created.name}


@app.post("/match")
def match(body: MatchIn) -> dict:
    return _match_or_404(body.vacancy_code)


@app.get("/match/{vacancy_code}")
def match_by_code(vacancy_code: str) -> dict:
    return _match_or_404(vacancy_code)
