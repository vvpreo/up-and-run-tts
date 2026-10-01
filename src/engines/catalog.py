"""Справочник моделей Silero: что можно загрузить, под какой лицензией и как с ними
обращаться. Это единственное место, где перечислены модели."""

from dataclasses import dataclass, field


@dataclass(frozen=True)
class ModelInfo:
    id: str
    title: str
    file: str  # имя файла на models.silero.ai/models/tts/ru/
    license: str
    commercial: bool
    # Встроенная расстановка ударений и омографов; иначе ударения ставит silero-stress
    builtin_stress: bool
    # Префикс голосов, которые говорят по-русски (в CIS-моделях много языков)
    voice_prefix: str = ""
    # Интонационные типы (type_str у apply_tts) — есть только у новых ru-моделей
    intonation: bool = False
    sample_rates: tuple = (8000, 24000, 48000)
    notes: str = ""
    tags: list = field(default_factory=list)


MODELS: dict[str, ModelInfo] = {
    m.id: m
    for m in [
        ModelInfo(
            id="v5_cis_base",
            title="Silero v5 CIS base",
            file="v5_cis_base.pt",
            license="MIT",
            commercial=True,
            builtin_stress=False,
            voice_prefix="ru_",
            notes="Голоса носителей языков СНГ, говорящие по-русски (слышен акцент). "
            "Ударения и «ё» ставит пакет silero-stress (MIT).",
        ),
        ModelInfo(
            id="v5_5_ru",
            title="Silero v5.5 ru",
            file="v5_5_ru.pt",
            license="CC BY-NC 4.0",
            commercial=False,
            builtin_stress=True,
            intonation=True,
            notes="Пять русских голосов, встроенные ударения, омографы и вопросительная интонация. "
            "Некоммерческая лицензия.",
        ),
        ModelInfo(
            id="v5_4_ru",
            title="Silero v5.4 ru",
            file="v5_4_ru.pt",
            license="CC BY-NC 4.0",
            commercial=False,
            builtin_stress=True,
            intonation=True,
            notes="Предыдущая версия v5 ru (без голоса eugene). Некоммерческая лицензия.",
        ),
        ModelInfo(
            id="v5_3_ru",
            title="Silero v5.3 ru",
            file="v5_3_ru.pt",
            license="CC BY-NC 4.0",
            commercial=False,
            builtin_stress=True,
            notes="Некоммерческая лицензия.",
        ),
        ModelInfo(
            id="v4_ru",
            title="Silero v4 ru",
            file="v4_ru.pt",
            license="CC BY-NC 4.0",
            commercial=False,
            builtin_stress=True,
            notes="Старое поколение; оставлено для сравнения. Некоммерческая лицензия.",
        ),
    ]
}

DEFAULT_MODEL = "v5_cis_base"
