"""Нарезка текста на фрагменты для синтеза: по предложениям всегда, по клаузам —
когда фрагмент достаточно длинный. Пунктуация между цифрами границей не считается.

Та же логика работает и инкрементально (IncrementalSplitter): текст приходит
кусками от LLM, фрагменты выдаются, как только граница подтверждена."""

import re

_PIECE = re.compile(r"[^.!?…;:,—\n]+[.!?…;:,—]*\n*\s*")
_HARD_END = re.compile(r"[.!?…]\s*$|\n\s*$")
_DIGIT_PUNCT = re.compile(r"(?<=\d)[.,:](?=\d)")
# Русские сокращения, после которых точка не конец предложения
_ABBREV = re.compile(r"(?:^|\s)(?:т|т\.е|т\.к|и\.т\.д|и\.т\.п|г|гг|ул|д|кв|им|см|стр|рис|руб|коп|тыс|млн|млрд|"
                     r"др|пр|напр|проф|акад|св|тел|ст|п|пп|гл|ч|р|с)\.$", re.I)
_GUARD = {".": "", ",": "", ":": ""}
_UNGUARD = {v: k for k, v in _GUARD.items()}


def _unguard(s: str) -> str:
    return "".join(_UNGUARD.get(ch, ch) for ch in s)


def split_fragments(text: str, min_fragment_chars: int = 25, min_tail: int = 15) -> list[str]:
    text = _DIGIT_PUNCT.sub(lambda m: _GUARD[m.group()], text)
    out, cur = [], ""
    total = len(text)
    consumed = 0
    for piece in _PIECE.findall(text):
        cur += piece
        consumed += len(piece)
        rest = total - consumed
        stripped = cur.strip()
        hard = bool(_HARD_END.search(cur)) and not _ABBREV.search(_unguard(stripped))
        if hard or (len(stripped) >= min_fragment_chars and rest >= min_tail):
            if stripped:
                out.append(_unguard(stripped))
            cur = ""
    if cur.strip():
        out.append(_unguard(cur.strip()))
    return out


class IncrementalSplitter:
    """Копит дельты текста и отдаёт готовые фрагменты. Последний (незакрытый)
    фрагмент выдаётся только по flush()."""

    def __init__(self, min_fragment_chars: int = 25, min_tail: int = 15, force_after_chars: int = 300):
        self.min_fragment_chars = min_fragment_chars
        self.min_tail = min_tail
        self.force_after_chars = force_after_chars
        self.buf = ""

    def push(self, delta: str) -> list[str]:
        self.buf += delta
        parts = split_fragments(self.buf, self.min_fragment_chars, self.min_tail)
        if len(parts) <= 1:
            if len(self.buf) >= self.force_after_chars and parts:
                self.buf = ""
                return parts
            return []
        # всё, кроме последнего куска, подтверждено границей
        done, last = parts[:-1], parts[-1]
        idx = self.buf.rfind(last)  # последний кусок — непустой суффикс буфера без пробелов по краям
        self.buf = self.buf[idx:] if idx >= 0 else last
        return done

    def flush(self) -> list[str]:
        parts = split_fragments(self.buf, self.min_fragment_chars, self.min_tail)
        self.buf = ""
        return parts
