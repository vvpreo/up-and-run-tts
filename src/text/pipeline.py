"""Конвейер подготовки текста к синтезу: разметка → нарезка → словарь →
числа → латиница → ударения → SSML. Один и тот же для запроса целиком и для
текста, приходящего кусками (там нарезчик инкрементальный, остальное — то же)."""

from dataclasses import dataclass
from xml.sax.saxutils import escape

from src.engines.catalog import ModelInfo
from src.text.markdown import strip_markdown
from src.text.normalize import normalize_latin, normalize_numbers
from src.text.splitter import split_fragments
from src.text.stress import accentor, apply_dictionary


@dataclass
class Prepared:
    text: str | None = None  # для apply_tts(text=...)
    ssml: str | None = None  # для apply_tts(ssml_text=...)
    source: str = ""  # фрагмент до обработки — для отладки и админки

    @property
    def shown(self) -> str:
        return self.ssml if self.ssml is not None else (self.text or "")


class TextPipeline:
    def __init__(self, text_cfg: dict, dictionary: dict[str, str], model: ModelInfo,
                 rate: str = "medium", pitch: str = "medium", put_yo: bool = True):
        self.cfg = text_cfg
        self.dictionary = dictionary
        self.model = model
        self.rate = rate
        self.pitch = pitch
        self.put_yo = put_yo

    # --- отдельный фрагмент -------------------------------------------------------
    def prepare_fragment(self, fragment: str) -> Prepared:
        text = fragment
        text = apply_dictionary(text, self.dictionary)
        if self.cfg.get("normalize_numbers", True):
            text = normalize_numbers(text)
        if self.cfg.get("normalize_latin", True):
            text = normalize_latin(text)
        if self.cfg.get("put_stress", True) and not self.model.builtin_stress:
            text = accentor(text, put_yo=self.put_yo)
        if self.rate != "medium" or self.pitch != "medium":
            ssml = f'<speak><prosody rate="{self.rate}" pitch="{self.pitch}">{escape(text)}</prosody></speak>'
            return Prepared(ssml=ssml, source=fragment)
        return Prepared(text=text, source=fragment)

    # --- весь текст ---------------------------------------------------------------
    def fragments(self, text: str) -> list[str]:
        if self.cfg.get("strip_markdown", True):
            text = strip_markdown(text)
        return split_fragments(text, int(self.cfg.get("min_fragment_chars", 25)))

    def prepare(self, text: str) -> list[Prepared]:
        return [self.prepare_fragment(f) for f in self.fragments(text)]


def is_ssml(text: str) -> bool:
    return text.lstrip().lower().startswith("<speak")


def rate_from_speed(speed: float | None, default: str) -> str:
    """OpenAI speed (0.25–4.0) → значение rate для SSML Silero (проценты)."""
    if speed is None or abs(speed - 1.0) < 1e-6:
        return default
    return f"{int(round(max(0.25, min(4.0, speed)) * 100))}%"
