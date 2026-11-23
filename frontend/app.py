"""Окно сопоставления резюме. Ходит в FastAPI, сам векторы не считает."""

from __future__ import annotations

import os

import flet as ft
import httpx

API_URL = os.environ.get("API_URL", "http://127.0.0.1:8000").rstrip("/")

VERDICT_COLOR = {
    "рекомендовать": ft.Colors.GREEN_700,
    "рассмотреть": ft.Colors.AMBER_800,
    "слабое соответствие": ft.Colors.BLUE_GREY_600,
}


def main(page: ft.Page) -> None:
    page.title = "Резюме и вакансии 1С"
    page.theme = ft.Theme(color_scheme_seed=ft.Colors.BLUE)
    page.theme_mode = ft.ThemeMode.LIGHT
    page.padding = 24
    page.scroll = ft.ScrollMode.AUTO
    window = getattr(page, "window", None)
    if window is not None:
        window.width = 1180
        window.height = 860

    vacancies: list[dict] = []
    status = ft.Text(size=13)
    vacancy_body = ft.Text(size=13, selectable=True)
    vacancy_skills = ft.Text(size=13, color=ft.Colors.BLUE_GREY_700)
    resume_list = ft.Column(spacing=2)
    results = ft.Column(spacing=10)
    vacancy_dd = ft.Dropdown(label="Вакансия", width=400, on_select=lambda e: show_vacancy())
    name_field = ft.TextField(label="Имя кандидата", width=400)
    text_field = ft.TextField(
        label="Текст резюме",
        multiline=True,
        min_lines=6,
        max_lines=10,
        width=400,
    )

    def show_vacancy() -> None:
        current = next((item for item in vacancies if item["code"] == vacancy_dd.value), None)
        if current is None:
            vacancy_body.value = ""
            vacancy_skills.value = ""
        else:
            vacancy_body.value = current["text"]
            skills = ", ".join(current["skills"]) or "—"
            vacancy_skills.value = f"Словарь вытащил: {skills}"
        page.update()

    def set_status(message: str, *, error: bool = False) -> None:
        status.value = message
        status.color = ft.Colors.RED_700 if error else ft.Colors.GREEN_800

    async def load_catalog(e: ft.ControlEvent | None = None) -> None:
        del e
        set_status("Читаю вакансии и резюме…")
        page.update()
        try:
            async with httpx.AsyncClient(timeout=30) as client:
                vacancy_response = await client.get(f"{API_URL}/vacancies")
                resume_response = await client.get(f"{API_URL}/resumes")
                vacancy_response.raise_for_status()
                resume_response.raise_for_status()
        except httpx.HTTPError as error:
            set_status(
                f"Бэкенд не ответил ({API_URL}). Запустите: uvicorn backend.app:app --port 8000. {error}",
                error=True,
            )
            page.update()
            return
        vacancies.clear()
        vacancies.extend(vacancy_response.json())
        vacancy_dd.options = [
            ft.dropdown.Option(key=item["code"], text=item["title"]) for item in vacancies
        ]
        if vacancies and vacancy_dd.value not in {item["code"] for item in vacancies}:
            vacancy_dd.value = vacancies[0]["code"]
        resume_list.controls = [
            ft.Text(f"• {item['name']}", size=13) for item in resume_response.json()
        ]
        show_vacancy()
        set_status(f"В индексе {len(resume_response.json())} резюме. Поиск идёт через Qdrant query_points.")
        page.update()

    async def find_candidates(e: ft.ControlEvent) -> None:
        del e
        if not vacancy_dd.value:
            set_status("Сначала выберите вакансию.", error=True)
            page.update()
            return
        set_status("Считаю близость…")
        page.update()
        try:
            async with httpx.AsyncClient(timeout=60) as client:
                response = await client.post(
                    f"{API_URL}/match",
                    json={"vacancy_code": vacancy_dd.value},
                )
                response.raise_for_status()
        except httpx.HTTPError as error:
            set_status(f"Не удалось сопоставить: {error}", error=True)
            page.update()
            return
        payload = response.json()
        rows = payload["results"]
        results.controls = [result_card(place, row) for place, row in enumerate(rows, start=1)]
        if not results.controls:
            results.controls = [ft.Text("Резюме в индексе нет.")]
        set_status(payload["formula"])
        page.update()

    async def add_resume(e: ft.ControlEvent) -> None:
        del e
        name = (name_field.value or "").strip()
        text = (text_field.value or "").strip()
        if not name or not text:
            set_status("Для нового резюме нужны имя и текст.", error=True)
            page.update()
            return
        set_status("Сохраняю файл и перестраиваю индекс…")
        page.update()
        try:
            async with httpx.AsyncClient(timeout=60) as client:
                response = await client.post(
                    f"{API_URL}/resumes",
                    json={"name": name, "text": text},
                )
                if response.status_code >= 400:
                    detail = response.json().get("detail", response.text)
                    set_status(str(detail), error=True)
                    page.update()
                    return
        except httpx.HTTPError as error:
            set_status(f"Не удалось сохранить резюме: {error}", error=True)
            page.update()
            return
        created = response.json()
        name_field.value = ""
        text_field.value = ""
        set_status(f"Добавлен {created['name']}. Файл лежит в resumes/{created['code']}.txt")
        await load_catalog()

    left = ft.Column(
        [
            ft.Text("Вакансия", size=18, weight=ft.FontWeight.W_600),
            vacancy_dd,
            vacancy_skills,
            vacancy_body,
            ft.Button("Найти кандидатов", on_click=find_candidates, width=220),
            ft.Divider(),
            ft.Text("Своё резюме", size=18, weight=ft.FontWeight.W_600),
            ft.Text(
                "Сохраняется в resumes/*.txt и сразу попадает в коллекцию Qdrant.",
                size=12,
                color=ft.Colors.BLUE_GREY_700,
            ),
            name_field,
            text_field,
            ft.Button("Добавить резюме", on_click=add_resume, width=220),
            ft.Divider(),
            ft.Text("Кто уже в индексе", size=16, weight=ft.FontWeight.W_600),
            resume_list,
        ],
        width=420,
        spacing=8,
    )
    right = ft.Column(
        [
            ft.Text("Результат query_points", size=18, weight=ft.FontWeight.W_600),
            ft.Text(
                "Порядок — по итоговому баллу. Место по вектору — как вернул Qdrant, до смеси с компетенциями.",
                size=12,
                color=ft.Colors.BLUE_GREY_700,
            ),
            results,
        ],
        expand=True,
        spacing=8,
    )
    page.add(
        ft.Text("Сопоставление резюме разработчиков 1С", size=26, weight=ft.FontWeight.W_600),
        ft.Text(f"Бэкенд {API_URL}", size=12, color=ft.Colors.BLUE_GREY_700),
        status,
        ft.Row(
            [left, right],
            vertical_alignment=ft.CrossAxisAlignment.START,
            spacing=28,
        ),
    )
    page.run_task(load_catalog)


def result_card(place: int, row: dict) -> ft.Container:
    color = VERDICT_COLOR.get(row["verdict"], ft.Colors.BLUE_GREY_700)
    matched = ", ".join(row["matched"]) or "—"
    missing = ", ".join(row["missing"]) or "—"
    terms = ", ".join(row["terms"]) or "—"
    return ft.Container(
        border=ft.Border.all(1, color),
        border_radius=12,
        padding=14,
        content=ft.Column(
            [
                ft.Row(
                    [
                        ft.Text(f"{place}. {row['name']}", size=18, weight=ft.FontWeight.W_600),
                        ft.Text(row["verdict"], color=color, weight=ft.FontWeight.W_600),
                    ],
                    alignment=ft.MainAxisAlignment.SPACE_BETWEEN,
                ),
                ft.Text(
                    f"балл {row['score']:.3f}    косинус {row['cosine']:.3f}    "
                    f"место по вектору {row['cosine_rank']}    покрытие {row['coverage']:.0%}"
                ),
                ft.Text(f"Совпало: {matched}", size=13),
                ft.Text(f"Пробелы: {missing}", size=13),
                ft.Text(f"Вклад терминов: {terms}", size=12, color=ft.Colors.BLUE_GREY_700),
            ],
            spacing=4,
        ),
    )


if __name__ == "__main__":
    ft.run(main)
