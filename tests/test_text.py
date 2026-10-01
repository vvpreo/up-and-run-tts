from src.engines.catalog import MODELS
from src.text.markdown import strip_markdown
from src.text.normalize import normalize_latin, normalize_numbers, ordinal, plural
from src.text.splitter import IncrementalSplitter, split_fragments
from src.text.stress import apply_dictionary


def test_numbers_dates_times_money():
    out = normalize_numbers("Встреча 12.05.2026 в 14:30, бюджет 1 250 000 руб., рост 3,5%")
    assert "двенадцатое мая две тысячи двадцать шестого года" in out
    assert "четырнадцать часов тридцать минут" in out
    assert "один миллион двести пятьдесят тысяч рублей" in out
    assert "три с половиной процента" in out


def test_units_and_agreement():
    assert normalize_numbers("21 рубль") == "двадцать один рубль"
    assert normalize_numbers("5 кг") == "пять килограммов"
    assert normalize_numbers("2 т") == "две тонны"
    assert normalize_numbers("1 мин") == "одна минута"
    assert normalize_numbers("в 2024 году") == "в две тысячи двадцать четвёртом году"
    assert normalize_numbers("3-я попытка") == "третья попытка"
    assert plural(11, "рубль", "рубля", "рублей") == "рублей"
    assert ordinal(2, "n") == "второе"


def test_unit_dot_keeps_sentence_boundary():
    out = normalize_numbers("Цена 100 руб. Дальше текст.")
    assert out == "Цена сто рублей. Дальше текст."
    assert normalize_numbers("100 руб., потом") == "сто рублей, потом"


def test_latin_rules_and_dictionary():
    assert normalize_latin("Python") == "питон"
    assert apply_dictionary("Установите Docker и GitHub", {}) == "Установите д+окер и гитх+аб"
    assert apply_dictionary("Мука", {"мука": "мук+а"}) == "мук+а"


def test_splitter_guards_digits_and_abbreviations():
    parts = split_fragments("Встреча 12.05.2026 в 14:30. Т.е. завтра, см. письмо. Бюджет 1,5 млн, это много, и это не всё.")
    assert parts[0] == "Встреча 12.05.2026 в 14:30."
    assert parts[1] == "Т.е. завтра, см. письмо."
    assert all(len(p) for p in parts)


def test_incremental_splitter_matches_batch():
    text = "Первое предложение, довольно длинное, с запятыми. Второе! Третье?"
    inc = IncrementalSplitter(min_fragment_chars=25)
    out = []
    for i in range(0, len(text), 3):
        out += inc.push(text[i:i + 3])
    out += inc.flush()
    # Разрезы могут отличаться (инкрементальный нарезчик не видит хвоста), но текст тот же
    assert " ".join(out) == " ".join(split_fragments(text, 25))
    assert out[-1] == "Третье?"


def test_markdown():
    out = strip_markdown("# Заголовок\n\n**Жирно** и `код`, [ссылка](http://x.y) 🙂\n\n```\nprint(1)\n```")
    assert out == "Заголовок\n\nЖирно и код, ссылка"


def test_catalog_license_flags():
    assert MODELS["v5_cis_base"].commercial and not MODELS["v5_cis_base"].builtin_stress
    assert not MODELS["v5_5_ru"].commercial and MODELS["v5_5_ru"].builtin_stress
