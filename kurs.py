"""Векторный анализ резюме и сопоставление компетенций кандидатов
с требованиями вакансий разработчиков 1С.

Учебный корпус в этом файле: пять вакансий и восемь резюме.
Свои резюме кладутся в папку resumes рядом со скриптом, по одному человеку на файл.
Первая строка — «Имя: Фамилия Имя», дальше пустая строка и текст.
Подхватываются все *.txt из этой папки, код менять не нужно.
Оценка пары «резюме — вакансия» складывается из двух независимых сигналов.

1. Векторная модель. Текст нормализуется, режется на слова и биграммы
   и переводится в TF-IDF. Близость — косинус между векторами.
   У векторов единичная длина, поэтому косинус равен скалярному произведению.
2. Словарь компетенций 1С. Из текста достаются навыки (в том числе
   по основе слова: «запросы» и «запросов» — один навык). Покрытие —
   доля требований вакансии, которые нашлись в резюме.

Итог: score = 0.6 * cosine + 0.4 * coverage.

Запуск из каталога bigdata:
    .venv/bin/python kurs/kurs.py
"""

from __future__ import annotations

import re
from dataclasses import dataclass
from pathlib import Path

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.metrics.pairwise import cosine_similarity

OUT_DIR = Path(__file__).resolve().parent
RESUMES_DIR = OUT_DIR / "resumes"
TEXT_WEIGHT = 0.6
SKILL_WEIGHT = 0.4
TOP_N = 3

# Служебные слова, которые есть почти в каждом объявлении и не отличают кандидата.
STOP_WORDS = [
    "и", "в", "во", "на", "с", "со", "по", "для", "от", "до", "не", "что",
    "это", "как", "а", "но", "или", "из", "за", "к", "ко", "о", "об", "при",
    "же", "бы", "то", "так", "его", "ее", "их", "мы", "вы", "он", "она", "они",
    "опыт", "работы", "работа", "работаю", "лет", "год", "года", "году",
    "требования", "требование", "знание", "знания", "умение", "навыки",
    "будет", "плюсом", "нужен", "нужны", "ищем", "команде", "уровне",
]


@dataclass(frozen=True)
class Skill:
    """Навык: точные фразы и основы слов (без окончания)."""

    title: str
    phrases: tuple[str, ...] = ()
    stems: tuple[str, ...] = ()


# Закрытый перечень того, что имеет смысл различать у разработчика 1С.
SKILLS: tuple[Skill, ...] = (
    Skill("Платформа 1С", phrases=("конфигуратор",), stems=("платформ",)),
    Skill("Встроенный язык", phrases=("встроенный язык", "встроенного языка")),
    Skill(
        "Язык запросов",
        phrases=("временные таблицы", "пакетные запросы"),
        stems=("запрос",),
    ),
    Skill("СКД", phrases=("скд", "система компоновки данных"), stems=("компоновк",)),
    Skill(
        "Управляемые формы",
        phrases=(
            "управляемые формы",
            "управляемая форма",
            "управляемых форм",
            "управляемыми формами",
        ),
    ),
    Skill("БСП", phrases=("бсп", "библиотека стандартных подсистем")),
    Skill("Расширения", stems=("расширен",)),
    Skill(
        "Обмен данными",
        phrases=(
            "обмен данными",
            "планы обмена",
            "план обмена",
            "конвертация данных",
            "enterprise data",
        ),
    ),
    Skill("УТ", phrases=("управление торговлей", "ут 11", "ут")),
    Skill("ERP", phrases=("erp", "ерп", "управление предприятием")),
    Skill("ЗУП", phrases=("зуп",), stems=("зарплат", "кадр")),
    Skill("Бухгалтерия", stems=("бухгалт",)),
    Skill(
        "Права и RLS",
        phrases=(
            "rls",
            "рлс",
            "роли доступа",
            "ограничение прав",
            "ограничение доступа на уровне записей",
        ),
    ),
    Skill(
        "Производительность",
        phrases=("технологический журнал", "apdex"),
        stems=("производительн", "оптимизац"),
    ),
    Skill(
        "Интеграции",
        phrases=(
            "http",
            "json",
            "xml",
            "rest",
            "odata",
            "soap",
            "веб-сервисы",
            "веб-сервис",
        ),
    ),
    Skill("SQL", phrases=("sql", "postgresql", "postgres", "субд")),
    Skill("Хранилище и Git", phrases=("git", "edt", "хранилище конфигурации")),
    Skill("Тестирование", phrases=("vanessa",), stems=("тестирован", "сценарн")),
    Skill(
        "Проведение документов",
        phrases=(
            "проведение документов",
            "проведения документов",
            "проведении документов",
        ),
    ),
)


@dataclass(frozen=True)
class Vacancy:
    code: str
    title: str
    text: str


@dataclass(frozen=True)
class Resume:
    code: str
    name: str
    text: str


VACANCIES: tuple[Vacancy, ...] = (
    Vacancy(
        "ut",
        "Middle 1С, УТ",
        """
        Ищем разработчика 1С уровня middle в команду внедрения Управление торговлей.
        Обязанности: доработка УТ 11, управляемые формы, проведение документов, печатные формы.
        Требования: платформа 1С, встроенный язык, язык запросов, временные таблицы, СКД,
        расширения конфигурации, отладка в конфигураторе.
        Будет плюсом: обмен данными, БСП, хранилище конфигурации.
        """,
    ),
    Vacancy(
        "erp",
        "Senior 1С, ERP",
        """
        Senior-разработчик 1С ERP. Сопровождение управления предприятием, сложные обмены,
        производительность. Требования: ERP, библиотека стандартных подсистем, язык запросов,
        оптимизация, технологический журнал, PostgreSQL, ограничение доступа на уровне записей,
        роли доступа, интеграции HTTP JSON REST, конвертация данных, планы обмена, EDT, Git.
        Нужен самостоятельный разработчик, который проводит код-ревью.
        """,
    ),
    Vacancy(
        "zup",
        "Разработчик ЗУП",
        """
        Разработчик 1С ЗУП 3.1. Расчет зарплаты, кадровые документы, отчеты
        на системе компоновки данных, управляемые формы, язык запросов,
        ограничение доступа на уровне записей, сценарное тестирование Vanessa.
        Нужен уверенный встроенный язык и опыт доработки типовой ЗУП.
        Платформа 1С, конфигуратор, проведение документов по кадрам.
        """,
    ),
    Vacancy(
        "jun",
        "Junior 1С",
        """
        Junior-разработчик 1С. Обучение в команде сопровождения бухгалтерии.
        Нужны базовые знания платформы 1С, встроенного языка, простых запросов
        и управляемых форм. Важно готовность разбираться в проведении документов.
        Коммерческий опыт не обязателен, конфигуратор на базовом уровне достаточен.
        """,
    ),
    Vacancy(
        "int",
        "Интегратор 1С",
        """
        Интегратор 1С. Обмен данными между 1С и внешними системами.
        Требования: HTTP, JSON, XML, REST, SOAP, OData, веб-сервисы, планы обмена,
        конвертация данных, Enterprise Data, обработка ошибок обмена, PostgreSQL.
        Встроенный язык нужен для модулей обмена. Отчеты для этой роли не требуются.
        """,
    ),
)


RESUMES: tuple[Resume, ...] = (
    Resume(
        "anna",
        "Анна Ковалёва",
        """
        Разработчик 1С, три года. Основная конфигурация — Управление торговлей, УТ 11.
        Делаю управляемые формы, проведение документов, печатные формы, расширения конфигурации.
        Хорошо знаю язык запросов и временные таблицы, отчеты на СКД, отладку в конфигураторе.
        Использую хранилище конфигурации. Немного дорабатывала обмен данными с сайтом.
        Платформа 1С, встроенный язык, БСП только на чтение чужого кода.
        """,
    ),
    Resume(
        "peter",
        "Пётр Соколов",
        """
        Ведущий разработчик 1С, шесть лет. Последние три года — ERP, управление предприятием.
        Проектирую подсистемы на БСП, сложные обмены, конвертация данных, планы обмена.
        Оптимизация запросов, технологический журнал, PostgreSQL, разбор блокировок.
        Интеграции по HTTP, JSON и REST. Роли доступа и ограничение доступа на уровне записей.
        Веду код-ревью, работаю в EDT и Git. Встроенный язык, платформа 1С, конфигуратор.
        """,
    ),
    Resume(
        "maria",
        "Мария Лебедева",
        """
        Разработчик 1С ЗУП, четыре года. ЗУП 3.1: расчет зарплаты, кадровые документы, начисления.
        Отчеты делаю на СКД, также управляемые формы, язык запросов и временные таблицы.
        Настраиваю роли доступа и RLS. Пишу сценарные тесты Vanessa Automation.
        Уверенный встроенный язык, платформа 1С, конфигуратор, проведение документов
        в типовой ЗУП.
        """,
    ),
    Resume(
        "igor",
        "Игорь Новиков",
        """
        Выпускник курсов по 1С. Учебные задачи: платформа 1С, встроенный язык,
        простые запросы, одна управляемая форма справочника.
        Смотрел проведение документов на демобазе бухгалтерии.
        Коммерческого опыта нет. Конфигуратор знаю на базовом уровне и готов учиться.
        """,
    ),
    Resume(
        "oleg",
        "Олег Васильев",
        """
        Интегратор 1С, пять лет. Обмен данными: HTTP, JSON, XML, REST, SOAP, OData, веб-сервисы.
        Планы обмена, конвертация данных, Enterprise Data, повторная отправка пакетов.
        Базы на PostgreSQL. Писал загрузки прайсов и статусов заказов.
        Встроенный язык использую для модулей обмена. Отчеты почти не делаю.
        Платформа 1С, конфигуратор.
        """,
    ),
    Resume(
        "sveta",
        "Светлана Морозова",
        """
        Бухгалтер. Ежедневно работаю в 1С Бухгалтерия: первичные документы, банк,
        закрытие месяца, ОСВ, декларации. Пользователь, не разработчик.
        Конфигурацию не меняю, запросы не пишу, формы не дорабатываю.
        Нужна стабильная типовая бухгалтерия без программирования.
        """,
    ),
    Resume(
        "dmitry",
        "Дмитрий Орлов",
        """
        Разработчик 1С, три с половиной года. УТ 11 и немного ERP.
        Управляемые формы, проведение документов, язык запросов, СКД, расширения.
        Делал обмен данными с CRM через HTTP и JSON, планы обмена.
        БСП использую точечно. Git знаю на уровне коммитов.
        Платформа 1С, встроенный язык, конфигуратор. Планы запросов к базе глубоко не разбирал.
        """,
    ),
    Resume(
        "elena",
        "Елена Павлова",
        """
        Аналитик 1С. Собираю требования пользователей, пишу техническое задание,
        ставлю задачи разработчикам. Проверяю готовые отчеты в ЗУП и бухгалтерии,
        сверяю обороты и начисления. Программный код не пишу и конфигурацию не меняю.
        Внедрение веду со стороны заказчика: протоколы, контрольные примеры, приемка.
        """,
    ),
)


@dataclass(frozen=True)
class PairScore:
    vacancy: Vacancy
    resume: Resume
    cosine: float
    coverage: float
    score: float
    matched: tuple[str, ...]
    missing: tuple[str, ...]
    terms: tuple[str, ...]

    @property
    def verdict(self) -> str:
        if self.score >= 0.62:
            return "рекомендовать"
        if self.score >= 0.40:
            return "рассмотреть"
        return "слабое соответствие"


def load_resume_file(path: Path) -> Resume:
    """Первая строка файла — имя, остальное — текст резюме."""
    raw = path.read_text(encoding="utf-8").strip()
    if not raw:
        raise ValueError(f"Пустой файл резюме: {path.name}")
    lines = raw.splitlines()
    name = re.sub(
        r"^(имя|фио|кандидат)\s*:\s*",
        "",
        lines[0].strip(),
        count=1,
        flags=re.IGNORECASE,
    ).strip()
    body = "\n".join(lines[1:]).strip()
    if not name or not body:
        raise ValueError(
            f"В {path.name} нужны имя на первой строке и текст резюме ниже"
        )
    return Resume(path.stem, name, body)


def load_resumes(folder: Path) -> tuple[Resume, ...]:
    """Все *.txt из папки. Нет папки — просто пустой список, учебный корпус остаётся."""
    if not folder.is_dir():
        return ()
    loaded: list[Resume] = []
    for path in sorted(folder.glob("*.txt")):
        loaded.append(load_resume_file(path))
    return tuple(loaded)


def normalize(text: str) -> str:
    """Нижний регистр, «ё» -> «е», латинская «c» в «1c», лишняя пунктуация снята."""
    text = text.lower().replace("ё", "е")
    text = re.sub(r"1c\b", "1с", text)
    text = re.sub(r"[^0-9a-zа-я+\s]", " ", text)
    return re.sub(r"\s+", " ", text).strip()


def _term_in(text: str, term: str, *, stem: bool) -> bool:
    if stem:
        pattern = rf"(?<![0-9a-zа-я]){re.escape(term)}"
    else:
        pattern = rf"(?<![0-9a-zа-я]){re.escape(term)}(?![0-9a-zа-я])"
    return re.search(pattern, text) is not None


def extract_skills(text: str) -> list[str]:
    """Навыки, чьи фразы или основы нашлись в нормализованном тексте."""
    normalized = normalize(text)
    found: list[str] = []
    for skill in SKILLS:
        hit = any(_term_in(normalized, phrase, stem=False) for phrase in skill.phrases)
        hit = hit or any(_term_in(normalized, stem, stem=True) for stem in skill.stems)
        if hit:
            found.append(skill.title)
    return found


def build_vectorizer(texts: list[str]) -> tuple[TfidfVectorizer, object]:
    """TF-IDF по униграммам и биграммам. sublinear_tf сглаживает повторы."""
    vectorizer = TfidfVectorizer(
        lowercase=False,  # normalize() уже привёл регистр
        token_pattern=r"(?u)\b[0-9a-zа-я]{2,}\b",
        ngram_range=(1, 2),
        stop_words=STOP_WORDS,
        sublinear_tf=True,
        norm="l2",
    )
    matrix = vectorizer.fit_transform(texts)
    return vectorizer, matrix


def contributing_terms(
    left_row: object,
    right_row: object,
    names: np.ndarray,
    limit: int = 6,
) -> tuple[str, ...]:
    """Термины, которые дают наибольший вклад в скалярное произведение."""
    product = left_row.multiply(right_row).tocoo()
    if product.nnz == 0:
        return ()
    order = np.argsort(product.data)[::-1][:limit]
    return tuple(
        f"{names[product.col[index]]} ({product.data[index]:.3f})" for index in order
    )


def score_pairs(
    vacancies: tuple[Vacancy, ...],
    resumes: tuple[Resume, ...],
    resume_matrix: object,
    vacancy_matrix: object,
    feature_names: np.ndarray,
) -> list[PairScore]:
    cosine = cosine_similarity(resume_matrix, vacancy_matrix)
    vacancy_skills = [extract_skills(item.text) for item in vacancies]
    resume_skills = [extract_skills(item.text) for item in resumes]

    pairs: list[PairScore] = []
    for i, resume in enumerate(resumes):
        for j, vacancy in enumerate(vacancies):
            required = vacancy_skills[j]
            have = set(resume_skills[i])
            matched = tuple(skill for skill in required if skill in have)
            missing = tuple(skill for skill in required if skill not in have)
            coverage = len(matched) / len(required) if required else 0.0
            cos = float(cosine[i, j])
            score = TEXT_WEIGHT * cos + SKILL_WEIGHT * coverage
            terms = contributing_terms(
                resume_matrix[i], vacancy_matrix[j], feature_names
            )
            pairs.append(
                PairScore(
                    vacancy=vacancy,
                    resume=resume,
                    cosine=cos,
                    coverage=coverage,
                    score=score,
                    matched=matched,
                    missing=missing,
                    terms=terms,
                )
            )
    return pairs


def pairs_frame(pairs: list[PairScore]) -> pd.DataFrame:
    frame = pd.DataFrame(
        [
            {
                "вакансия": pair.vacancy.title,
                "кандидат": pair.resume.name,
                "косинус": round(pair.cosine, 3),
                "покрытие": round(pair.coverage, 3),
                "балл": round(pair.score, 3),
                "решение": pair.verdict,
                "совпало": "; ".join(pair.matched),
                "пробелы": "; ".join(pair.missing),
                "термины": "; ".join(pair.terms),
            }
            for pair in pairs
        ]
    )
    return frame.sort_values(["вакансия", "балл"], ascending=[True, False]).reset_index(
        drop=True
    )


def print_method(vectorizer: TfidfVectorizer, sample_count: int) -> None:
    names = vectorizer.get_feature_names_out()
    idf = pd.Series(vectorizer.idf_, index=names)
    probes = [
        token
        for token in ("скд", "зуп", "erp", "json", "vanessa", "запросов", "платформа")
        if token in idf.index
    ]
    print("Векторная модель TF-IDF")
    print(f"  документов в корпусе: {sample_count}")
    print(f"  терминов в словаре:   {len(names)}")
    print("  tf заменяется на 1 + ln(tf), чтобы десять повторов слова не задавили остальные")
    print("  idf = ln((1 + N) / (1 + df)) + 1, затем вектор делится на свою длину")
    print("  косинус таких векторов равен их скалярному произведению")
    print(f"  итог = {TEXT_WEIGHT:.1f} * косинус + {SKILL_WEIGHT:.1f} * покрытие компетенций")
    if probes:
        shown = ", ".join(f"{token}={idf[token]:.2f}" for token in probes)
        print(f"  примеры idf: {shown}")
    print()


def print_ranking(pairs: list[PairScore]) -> None:
    for vacancy in VACANCIES:
        ranked = sorted(
            (pair for pair in pairs if pair.vacancy.code == vacancy.code),
            key=lambda pair: pair.score,
            reverse=True,
        )
        print(f"Вакансия: {vacancy.title}")
        for place, pair in enumerate(ranked[:TOP_N], start=1):
            matched = ", ".join(pair.matched) or "—"
            missing = ", ".join(pair.missing) or "—"
            terms = ", ".join(pair.terms) or "—"
            print(
                f"  {place}. {pair.resume.name}: балл {pair.score:.3f} "
                f"({pair.verdict})"
            )
            print(
                f"     косинус {pair.cosine:.3f}, покрытие {pair.coverage:.0%} "
                f"({len(pair.matched)}/{len(pair.matched) + len(pair.missing)})"
            )
            print(f"     совпало: {matched}")
            print(f"     пробелы: {missing}")
            print(f"     вклад терминов: {terms}")
        print()


def print_channel_gap(pairs: list[PairScore]) -> None:
    """Пара, где словарь и TF-IDF расходятся сильнее всего."""
    gap_pair = max(pairs, key=lambda pair: abs(pair.cosine - pair.coverage))
    delta = gap_pair.coverage - gap_pair.cosine
    print("Где каналы расходятся")
    print(
        f"  {gap_pair.resume.name} / {gap_pair.vacancy.title}: "
        f"покрытие {gap_pair.coverage:.0%}, косинус {gap_pair.cosine:.3f}, "
        f"разница {delta:+.3f}"
    )
    if delta >= 0:
        print(
            "  Покрытие выше косинуса: словарь сводит падежи и сокращения в один навык, "
            "а TF-IDF оставляет «управляемая форма» и «управляемых форм» разными координатами. "
            "Редкие слова только одного текста дополнительно разводят векторы."
        )
    else:
        print(
            "  Косинус выше покрытия: общие слова сблизили векторы, "
            "но часть требований вакансии в резюме так и не названа."
        )
    print()
    print("Ограничения модели")
    print("  нет лемматизации: «запросы» и «запросов» — разные координаты вектора;")
    print("  отрицание не читается: у Светланы Морозовой «запросы не пишу» всё равно зачитывается как навык;")
    print("  словарь замкнут: навык вне списка попадает только в косинус, если совпали слова;")
    print("  упоминание навыка не равно владению им, уровень по тексту не восстановить;")
    print("  корпус учебный и маленький, поэтому idf редких слов завышен.")
    print()


def save_table(frame: pd.DataFrame) -> Path:
    path = OUT_DIR / "kurs_results.csv"
    frame.to_csv(path, index=False, encoding="utf-8-sig")
    return path


def save_plots(pairs: list[PairScore], resumes: tuple[Resume, ...]) -> tuple[Path, Path]:
    plt.rcParams.update(
        {
            "font.family": "DejaVu Sans",
            "axes.titlesize": 13,
            "axes.labelsize": 11,
        }
    )
    score = np.array(
        [
            [
                next(
                    pair.score
                    for pair in pairs
                    if pair.resume.code == resume.code and pair.vacancy.code == vacancy.code
                )
                for vacancy in VACANCIES
            ]
            for resume in resumes
        ]
    )

    heat_path = OUT_DIR / "kurs_heatmap.png"
    fig, ax = plt.subplots(figsize=(10.2, max(6.4, 0.48 * len(resumes) + 2)))
    image = ax.imshow(score, cmap="YlGnBu", vmin=0, vmax=1)
    ax.set_xticks(range(len(VACANCIES)), [item.title for item in VACANCIES], rotation=20, ha="right")
    ax.set_yticks(range(len(resumes)), [item.name for item in resumes])
    ax.set_title("Итоговый балл: резюме и вакансии 1С")
    for row in range(score.shape[0]):
        for col in range(score.shape[1]):
            color = "white" if score[row, col] >= 0.62 else "black"
            ax.text(col, row, f"{score[row, col]:.2f}", ha="center", va="center", color=color, fontsize=8)
    fig.colorbar(image, ax=ax, label="0.6 · косинус + 0.4 · покрытие")
    fig.tight_layout()
    fig.savefig(heat_path, dpi=140)
    plt.close(fig)

    best = []
    for vacancy in VACANCIES:
        best.append(
            max(
                (pair for pair in pairs if pair.vacancy.code == vacancy.code),
                key=lambda pair: pair.score,
            )
        )

    rank_path = OUT_DIR / "kurs_ranking.png"
    fig, ax = plt.subplots(figsize=(10.2, 5.6))
    x = np.arange(len(best))
    width = 0.36
    ax.bar(x - width / 2, [pair.cosine for pair in best], width, label="Косинус TF-IDF", color="#2c7fb8")
    ax.bar(
        x + width / 2,
        [pair.coverage for pair in best],
        width,
        label="Покрытие компетенций",
        color="#7fcdbb",
    )
    ax.set_ylim(0, 1.05)
    ax.set_ylabel("Значение")
    ax.set_title("Лучший кандидат по каждой вакансии")
    ax.set_xticks(
        x,
        [f"{pair.vacancy.title}\n{pair.resume.name.split()[0]}" for pair in best],
    )
    ax.legend(frameon=False, ncol=2, loc="upper center", bbox_to_anchor=(0.5, -0.28))
    fig.tight_layout()
    fig.subplots_adjust(bottom=0.28)
    fig.savefig(rank_path, dpi=140)
    plt.close(fig)
    return heat_path, rank_path


def main() -> None:
    from_files = load_resumes(RESUMES_DIR)
    resumes = RESUMES + from_files
    resume_texts = [normalize(item.text) for item in resumes]
    vacancy_texts = [normalize(item.text) for item in VACANCIES]
    vectorizer, matrix = build_vectorizer(resume_texts + vacancy_texts)
    resume_matrix = matrix[: len(resumes)]
    vacancy_matrix = matrix[len(resumes) :]
    feature_names = vectorizer.get_feature_names_out()

    pairs = score_pairs(
        VACANCIES, resumes, resume_matrix, vacancy_matrix, feature_names
    )
    frame = pairs_frame(pairs)

    print("Курсовая: векторный анализ резюме разработчиков 1С\n")
    print(f"Свои резюме: положите *.txt в {RESUMES_DIR}")
    print("Формат файла: первая строка «Имя: Фамилия Имя», затем текст.")
    if from_files:
        loaded = ", ".join(item.name for item in from_files)
        print(f"Сейчас прочитано из файлов: {loaded}")
    else:
        print("Файлов *.txt в папке нет, считаются только резюме из кода.")
    print()
    print_method(vectorizer, matrix.shape[0])
    print_ranking(pairs)
    print_channel_gap(pairs)

    short = frame[["вакансия", "кандидат", "косинус", "покрытие", "балл", "решение"]]
    print("Полная таблица")
    print(short.to_string(index=False))
    print()

    table_path = save_table(frame)
    heat_path, rank_path = save_plots(pairs, resumes)
    print(f"Таблица:  {table_path.name}")
    print(f"Тепловая карта: {heat_path.name}")
    print(f"Лучшие пары:    {rank_path.name}")


if __name__ == "__main__":
    main()
