"""Нормализация русского текста перед синтезом: Silero молча пропускает цифры и
латиницу, поэтому числа, даты, время, проценты, деньги и единицы раскрываются в
слова здесь. Правила — не морфологический анализатор: падеж числительных
подбирается только там, где он предсказуем (даты, деньги, единицы)."""

import re

from num2words import num2words

MONTHS_GEN = ["января", "февраля", "марта", "апреля", "мая", "июня", "июля", "августа", "сентября", "октября",
              "ноября", "декабря"]


def cardinal(n: int | float | str) -> str:
    return num2words(n, lang="ru")


def ordinal(n: int, gender: str = "m", case: str = "nom") -> str:
    """Порядковое: род m/f/n, падеж nom/gen. Меняется только последнее слово."""
    words = num2words(n, lang="ru", to="ordinal").split()
    last = words[-1]
    endings = {
        ("m", "nom"): None,
        ("f", "nom"): {"ый": "ая", "ой": "ая", "ий": "ья"},
        ("n", "nom"): {"ый": "ое", "ой": "ое", "ий": "ье"},
        ("m", "gen"): {"ый": "ого", "ой": "ого", "ий": "ьего"},
        ("n", "gen"): {"ый": "ого", "ой": "ого", "ий": "ьего"},
        ("f", "gen"): {"ый": "ой", "ой": "ой", "ий": "ьей"},
    }[(gender, case)]
    if endings:
        for src, dst in endings.items():
            if last.endswith(src):
                last = last[: -len(src)] + dst
                break
    words[-1] = last
    return " ".join(words)


def plural(n: int, one: str, few: str, many: str) -> str:
    """рубль / рубля / рублей по числу."""
    n = abs(n) % 100
    if 11 <= n <= 19:
        return many
    n %= 10
    if n == 1:
        return one
    if 2 <= n <= 4:
        return few
    return many


UNITS = {
    "руб": ("рубль", "рубля", "рублей"), "₽": ("рубль", "рубля", "рублей"), "р": ("рубль", "рубля", "рублей"),
    "коп": ("копейка", "копейки", "копеек"),
    "$": ("доллар", "доллара", "долларов"), "€": ("евро", "евро", "евро"),
    "%": ("процент", "процента", "процентов"),
    "кг": ("килограмм", "килограмма", "килограммов"), "г": ("грамм", "грамма", "граммов"),
    "т": ("тонна", "тонны", "тонн"), "км": ("километр", "километра", "километров"),
    "м": ("метр", "метра", "метров"), "см": ("сантиметр", "сантиметра", "сантиметров"),
    "мм": ("миллиметр", "миллиметра", "миллиметров"), "л": ("литр", "литра", "литров"),
    "мл": ("миллилитр", "миллилитра", "миллилитров"), "ч": ("час", "часа", "часов"),
    "мин": ("минута", "минуты", "минут"), "сек": ("секунда", "секунды", "секунд"), "с": ("секунда", "секунды", "секунд"),
    "шт": ("штука", "штуки", "штук"), "чел": ("человек", "человека", "человек"),
    "гб": ("гигабайт", "гигабайта", "гигабайт"), "мб": ("мегабайт", "мегабайта", "мегабайт"),
    "кб": ("килобайт", "килобайта", "килобайт"), "тб": ("терабайт", "терабайта", "терабайт"),
    "°c": ("градус", "градуса", "градусов"), "°": ("градус", "градуса", "градусов"),
}
SCALES = {"тыс": 1000, "млн": 1_000_000, "млрд": 1_000_000_000}

_NUM = r"\d+(?:[  ]\d{3})*(?:[.,]\d+)?"


def _parse(num: str) -> tuple[int | float, bool]:
    s = num.replace(" ", "").replace(" ", "")
    if "," in s or "." in s:
        return float(s.replace(",", ".")), True
    return int(s), False


def _number_words(value, is_float: bool, gender: str = "m") -> str:
    if is_float:
        if value == int(value):
            value, is_float = int(value), False
        else:
            frac = round(value - int(value), 6)
            if abs(frac - 0.5) < 1e-9:
                return f"{cardinal(int(value))} с половиной" if int(value) else "половина"
            return cardinal(str(value))
    words = cardinal(value)
    if gender == "f":
        words = re.sub(r"\bодин$", "одна", words)
        words = re.sub(r"\bдва$", "две", words)
    return words


def _with_unit(value, is_float: bool, forms: tuple[str, str, str], feminine: bool) -> str:
    if is_float and value != int(value):
        return f"{_number_words(value, True)} {forms[1]}"
    n = int(value)
    return f"{_number_words(n, False, 'f' if feminine else 'm')} {plural(n, *forms)}"


def _is_feminine(forms) -> bool:
    return forms[0].endswith(("а", "я")) and forms[0] not in ("мужчина",)


_DATE = re.compile(r"\b(\d{1,2})\.(\d{1,2})\.(\d{4})\b")
_DATE_SHORT = re.compile(r"\b(\d{1,2})\.(\d{1,2})\b(?!\.\d)")
_TIME = re.compile(r"\b(\d{1,2}):(\d{2})(?::\d{2})?\b")
_YEAR = re.compile(r"\b(\d{4})\s*(?P<word>г\.|году|года|годом|годов|годах|год)(?![а-яё])", re.I)
_ORDINAL = re.compile(r"\b(\d+)-?(го|му|й|я|е|ой|ом|ых|ми|х)\b")
_SCALE_UNIT = re.compile(
    rf"(?P<num>{_NUM})\s*(?P<scale>тыс|млн|млрд)\.?\s*(?P<unit>руб|р|₽|\$|€|%|коп)?(?:\.(?=\s*(?-i:[а-яё0-9])|[,;:]))?(?![а-яё])", re.I)
_UNIT = re.compile(
    rf"(?P<num>{_NUM})\s*(?P<unit>руб|коп|кг|км|мм|см|мл|мб|гб|кб|тб|мин|сек|шт|чел|°c|°|₽|\$|€|%|[гтмлчср])(?:\.(?=\s*(?-i:[а-яё0-9])|[,;:]))?(?![а-яё])",
    re.I)
_RANGE = re.compile(rf"({_NUM})\s*[-–—]\s*({_NUM})")
_PLAIN = re.compile(_NUM)
_LATIN_WORD = re.compile(r"[A-Za-z][A-Za-z'’\-]*")


def _date(m: re.Match) -> str:
    d, mo, y = int(m.group(1)), int(m.group(2)), int(m.group(3))
    if not (1 <= d <= 31 and 1 <= mo <= 12):
        return m.group(0)
    return f"{ordinal(d, 'n')} {MONTHS_GEN[mo - 1]} {ordinal(y, 'm', 'gen')} года"


def _date_short(m: re.Match) -> str:
    d, mo = int(m.group(1)), int(m.group(2))
    if not (1 <= d <= 31 and 1 <= mo <= 12):
        return m.group(0)
    return f"{ordinal(d, 'n')} {MONTHS_GEN[mo - 1]}"


def _time(m: re.Match) -> str:
    h, mi = int(m.group(1)), int(m.group(2))
    if h > 23 or mi > 59:
        return m.group(0)
    hours = f"{cardinal(h)} {plural(h, 'час', 'часа', 'часов')}"
    if mi == 0:
        return f"{hours} ровно"
    minutes = f"ноль {cardinal(mi)}" if mi < 10 else cardinal(mi)
    return f"{hours} {minutes} {plural(mi, 'минута', 'минуты', 'минут')}"


def _year(m: re.Match) -> str:
    word = m.group("word").lower()
    n = int(m.group(1))
    if word == "год":
        return f"{ordinal(n)} год"
    if word == "году":  # предложный: в две тысячи двадцать четвёртом году
        return f"{ordinal(n, 'm', 'gen')[:-3]}ом году" if ordinal(n, 'm', 'gen').endswith("ого") else f"{ordinal(n, 'm', 'gen')} году"
    return f"{ordinal(n, 'm', 'gen')} {'года' if word == 'г.' else word}"


def _ordinal_suffix(m: re.Match) -> str:
    n, suf = int(m.group(1)), m.group(2)
    table = {"го": ("m", "gen"), "му": ("m", "gen"), "й": ("m", "nom"), "я": ("f", "nom"), "е": ("n", "nom"),
             "ой": ("f", "gen"), "ом": ("m", "gen"), "ых": ("m", "gen"), "ми": ("m", "gen"), "х": ("m", "gen")}
    g, c = table[suf]
    return ordinal(n, g, c)


def _scale_unit(m: re.Match) -> str:
    value, is_float = _parse(m.group("num"))
    value = value * SCALES[m.group("scale").lower()]
    unit = (m.group("unit") or "").lower()
    if unit:
        return _with_unit(value, is_float, UNITS[unit], _is_feminine(UNITS[unit]))
    return _number_words(value, is_float)


def _unit(m: re.Match) -> str:
    value, is_float = _parse(m.group("num"))
    forms = UNITS[m.group("unit").lower()]
    return _with_unit(value, is_float, forms, _is_feminine(forms))


def _range(m: re.Match) -> str:
    a, fa = _parse(m.group(1))
    b, fb = _parse(m.group(2))
    return f"{_number_words(a, fa)} — {_number_words(b, fb)}"


def _plain(m: re.Match) -> str:
    value, is_float = _parse(m.group(0))
    return _number_words(value, is_float)


def normalize_numbers(text: str) -> str:
    text = _DATE.sub(_date, text)
    text = _TIME.sub(_time, text)
    text = _YEAR.sub(_year, text)
    text = _DATE_SHORT.sub(_date_short, text)
    text = _ORDINAL.sub(_ordinal_suffix, text)
    text = _SCALE_UNIT.sub(_scale_unit, text)
    text = _UNIT.sub(_unit, text)
    text = _RANGE.sub(_range, text)
    text = _PLAIN.sub(_plain, text)
    return text


# --- латиница ----------------------------------------------------------------------
# Чтение латинских слов «по-английски» грубыми правилами. Точное произношение
# терминов задаётся в словаре произношений — он применяется раньше.
_DIGRAPHS = [
    ("tion", "шн"), ("sion", "жн"), ("ough", "у"), ("igh", "ай"), ("ck", "к"), ("ch", "ч"), ("sh", "ш"), ("th", "т"),
    ("ph", "ф"), ("wh", "в"), ("qu", "кв"), ("ee", "и"), ("ea", "и"), ("oo", "у"), ("ou", "ау"), ("ow", "оу"),
    ("ai", "эй"), ("ay", "эй"), ("ey", "ей"), ("oa", "оу"), ("ie", "и"), ("ei", "ей"), ("ng", "нг"), ("kn", "н"),
    ("wr", "р"), ("x", "кс"), ("ya", "я"), ("yu", "ю"), ("yo", "ё"), ("ye", "е"),
]
_LETTERS = {"a": "а", "b": "б", "c": "к", "d": "д", "e": "е", "f": "ф", "g": "г", "h": "х", "i": "и", "j": "дж",
            "k": "к", "l": "л", "m": "м", "n": "н", "o": "о", "p": "п", "r": "р", "s": "с", "t": "т", "u": "у",
            "v": "в", "w": "в", "y": "и", "z": "з"}
_LETTER_NAMES = {"a": "эй", "b": "би", "c": "си", "d": "ди", "e": "и", "f": "эф", "g": "джи", "h": "эйч", "i": "ай",
                 "j": "джей", "k": "кей", "l": "эл", "m": "эм", "n": "эн", "o": "оу", "p": "пи", "q": "кью",
                 "r": "ар", "s": "эс", "t": "ти", "u": "ю", "v": "ви", "w": "дабл-ю", "x": "икс", "y": "уай", "z": "зед"}


def transliterate_word(word: str) -> str:
    w = word.lower().replace("’", "").replace("'", "")
    if word.isupper() and 2 <= len(w) <= 4:  # аббревиатура: по буквам
        return " ".join(_LETTER_NAMES.get(ch, ch) for ch in w if ch.isalpha())
    out = ""
    i = 0
    while i < len(w):
        for src, dst in _DIGRAPHS:
            if w.startswith(src, i):
                out += dst
                i += len(src)
                break
        else:
            ch = w[i]
            if ch == "c" and i + 1 < len(w) and w[i + 1] in "eiy":
                out += "с"
            elif ch == "e" and i == len(w) - 1 and len(w) > 2:
                pass  # немое e на конце: make, code
            else:
                out += _LETTERS.get(ch, ch if ch.isalpha() else "")
            i += 1
    return out


def normalize_latin(text: str) -> str:
    return _LATIN_WORD.sub(lambda m: transliterate_word(m.group(0)), text)
