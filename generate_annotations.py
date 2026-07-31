import unicodedata
from collections import Counter
from typing import List, Dict, Tuple, Optional
import json
import re
# import gc
from pathlib import Path

import pandas as pd
# import torch
# from tqdm import tqdm
# from vllm import LLM, SamplingParams
# from vllm.sampling_params import StructuredOutputsParams
from jsonargparse import CLI
from rapidfuzz import distance, process

# Qwen/Qwen2.5-7B-Instruct-AWQ bs 192

JSON_SCHEMA = {
    "type": "object",
    "properties": {
        "setup": {"type": "string"},
        "punchline": {"type": "string"},
        "theme": {"type": "array", "items": {"type": "string"}},
        "mechanism": {"type": "array", "items": {"type": "string"}},
        "actors": {"type": "array", "items": {"type": "string"}}
    },
    "required": ["setup", "punchline", "theme", "mechanism", "actors"]
}

THEME_CLASSES = [
    # --- Работа и профессии ---
    "работа", "офис", "начальство", "подчинённые",
    "программисты", "врачи", "учителя", "полицейские",
    "военные", "юристы", "бухгалтеры", "строители",
    "водители", "продавцы", "инженеры", "учёные",
    "журналисты", "артисты", "музыканты", "писатели",
    "художники", "спортсмены", "повара", "шахтёры",
    "моряки", "лётчики", "пожарные", "сантехники",
    "фриланс", "безработица", "зарплата", "премия",
    "отпуск", "командировка", "собеседование", "увольнение", "пенсия",
    
    # --- Семья и отношения ---
    "семья", "брак", "свадьба", "развод",
    "муж", "жена", "дети", "подростки", "родители",
    "тёща", "свекровь", "зять", "невестка",
    "бабушка", "дедушка", "внуки",
    "любовь", "измена", "ревность", "свидание", "секс", "мать", "отец", "дочь", "сын",
    
    # --- Образование ---
    "школа", "университет", "студенты", "экзамены",
    "учёба", "профессора", "диплом",
    
    # --- Здоровье ---
    "медицина", "больница", "пациенты", "лекарства",
    "болезни", "психиатрия", "алкоголь", "курение",
    "спорт", "диета",
    
    # --- Политика и общество ---
    "политика", "власть", "правительство", "президент",
    "депутаты", "выборы", "законы", "суд",
    "армия", "тюрьма", "коррупция",
    "война", "экономика", "кризис", "налоги",
    
    # --- Религия ---
    "религия", "бог", "церковь", "священники",
    "молитва", "грех", "рай", "ад", "атеизм",
    
    # --- Национальности (классика русского анекдота) ---
    "русские", "украинцы", "американцы", "англичане",
    "французы", "немцы", "итальянцы", "евреи",
    "китайцы", "японцы", "армяне", "грузины",
    "татары", "чукчи", "эстонцы",
    
    # --- Быт и повседневность ---
    "быт", "дом", "квартира", "ремонт",
    "еда", "напитки", "одежда", "мода",
    "уборка", "готовки", "покупки", "магазины",
    "транспорт", "метро", "такси", "поезд", "самолёт",
    "машина", "дорога", "пробки",
    "дача", "огород", "деревня", "город",
    "путешествия", "туризм", "пляж", "море",
    "рыбалка", "охота",
    
    # --- Деньги ---
    "деньги", "богатство", "бедность", "кредит",
    "долг", "наследство", "лотерея", "бизнес",
    
    # --- Технологии ---
    "компьютеры", "интернет", "смартфон",
    "соцсети", "искусственный интеллект", "космос", "наука",
    
    # --- Животные ---
    "животные", "собаки", "кошки", "лошади",
    "медведи", "волки", "лисы", "зайцы", "рыбы", "птицы",
    
    # --- Время и события ---
    "время", "прошлое", "будущее",
    "понедельник", "пятница", "выходные",
    "праздники", "новый_год", "день рождения",
    
    # --- Смерть ---
    "смерть", "похороны", "кладбище",
    
    # --- Спорт ---
    "футбол", "хоккей", "шахматы", "олимпиада",
    
    # --- Культура ---
    "искусство", "музыка", "кино", "театр",
    "литература", "книги", "поэзия",
    
    # --- Философия и жизнь ---
    "жизнь", "счастье", "горе", "мудрость", "глупость",
    "истина", "ложь", "добро", "зло", "свобода", "судьба",
    
    # --- Специфические тематики ---
    "абсурд", "повседневность", "необитаемый остров",
    "русский немецкий американец", "штирлиц", "поручик рябов",
    "рабинович", "муж жена любовник",

    "бисексуалы", "геи", "трансгендеры",
    "чёрный юмор",
    "женщины",
    "мужчины",
    "криминал",
    "образование",
    "молодёжь",
    "старость",
    "дружба",
    "путешествия",
    "праздники",
]

MECHANISM_CLASSES = [
    # --- Лингвистические механизмы ---
    "каламбур",                    # Игра слов на созвучии
    "игра слов",                   # Многозначность слов
    "омонимия",                    # Одинаковое написание, разный смысл
    "паронимия",                   # Близкое звучание (адресат/адресант)
    "двусмысленность",             # Двойной смысл фразы
    "многозначность",              # Несколько значений слова
    "ирония",                      # Скрытая насмешка
    "сарказм",                     # Жёсткая ирония
    "гипербола",                   # Намеренное преувеличение
    "литота",                      # Намеренное преуменьшение
    "метафора",                    # Скрытое сравнение
    "сравнение",                   # Явное сопоставление
    "аллюзия",                     # Отсылка к известному
    "оксюморон",                   # Сочетание противоположностей
    "эвфемизм",                    # Мягкая замена грубого
    
    # --- Когнитивные механизмы ---
    "обман ожидания",              # Нарушение прогноза читателя
    "рефрейминг",                  # Смена точки зрения
    "инверсия",                    # Перевёртывание смысла
    "абсурд",                      # Логически невозможная ситуация
    "нонсенс",                     # Полная бессмыслица
    "алогизм",                     # Нарушение логики
    "парадокс",                    # Противоречивое, но верное
    "подмена понятий",             # Замена одного понятия другим
    "смещение фокуса",             # Перенос внимания
    "буквальная интерпретация",    # Понимание фразы буквально
    "ложный силлогизм",            # Ошибочный логический вывод
    "софизм",                      # Намеренная логическая ошибка
    "редукция до абсурда",         # Доведение идеи до нелепости
    
    # --- Структурные механизмы ---
    "правило трёх",                # Трёхчастная структура
    "антиклимакс",                 # Ослабление к концу
    "бафос",                       # Контраст высокого и низкого
    "комическое повторение",       # Повтор с вариацией
    "неожиданная развязка",        # Surprise ending
    "комическое преувеличение",    # Юмористическая гипербола
    "комическое противопоставление", # Контраст для юмора
    "комическое совпадение",       # Невероятное совпадение
    "эффект неожиданности",        # Surprise effect
    "комическая ситуация",         # Смешная ситуация
    "комический персонаж",         # Смешной герой
    "комический диалог",           # Смешной разговор
    "комическая деталь",           # Смешная подробность
    "комический тайминг",          # Timing в шутке
    "сюжетный поворот",            # Twist
    "инверсия ролей",              # Обмен ролями
    
    # --- Жанровые/тематические ---
    "чёрный юмор",                 # Юмор о смерти/трагедиях
    "пошлость",                    # Вульгарный юмор
    "сатира",                      # Обличительный юмор
    "самоирония",                  # Юмор над собой
    "гротеск",                     # Фантастическое преувеличение
    "карикатура",                  # Преувеличенное изображение
    "пародия",                     # Комическое подражание
    "стёб",                        # Издевательская ирония
    "идиотизм",                    # Юмор на глупости
    "тавтология",                  # "Масло масляное"
    "ответ на вопрос",
    "цитирование",
]

SYSTEM_PROMPT =  f'\
Ты — эксперт по лингвистике и структуре юмора. Твоя задача — проанализировать анекдот и разбить его на структурные части. Верни ответ строго в формате JSON.\
Поля для ответа:\
"setup": Завязка. Контекст, персонажи, ситуация. Без кульминации.\
"punchline": Развязка. Собственно шутка, игра слов или неожиданный поворот.\
"theme": Нужно выбрать минимальный, максимально полно описывающий текст набор тематик строго из перечисленных далее: {THEME_CLASSES}.\
"mechanism": Нужно выбрать минимальный, максимально полно описывающий текст набор механизмов анекдота строго из перечисленных далее: {MECHANISM_CLASSES}.\
"actors": Персонажи (например, "штирлиц", "русский, немец и американец", "депутат", "программист", "улитка"). Является списком строк. Может быть пустым списком, если персонажи \
не указаны явно. Если это не какие-то известные личности или персонажи, то стоит их обозначить обобщенной группой. например, "программист(ы)", "шахтер(ы)", "дети".\
Если анекдот слишком короткий и не делится, помести всё в "setup", а в "punchline" напиши оставь пустую строку "".\
Конкатенация полей "setup" и "punchline" должна быть строго равна исходному тексту, с точностью до символа.\
Пример текста: "Неожиданно найденный клад сорвал похороны"\
Пример ожидаемого ответа:\n' + '{\
    "setup": "Неожиданно найденный клад",\
    "punchline": " сорвал похороны",\
    "theme": ["абсурд"],\
    "mechanism" : ["черный юмор"],\
    "actors": [],\
}\
Пример 2. Текст: "У каждой домохозяйки есть свой маленький секретик. Надежда Константиновна, например, выводит пятна уксусом, а Татьяна Андреевна отравила своего мужа."\
Ответ:\
{\
    "setup": "У каждой домохозяйки есть свой маленький секретик. Надежда Константиновна, например, выводит пятна уксусом",\
    "punchline": ", а Татьяна Андреевна отравила своего мужа.",\
    "theme": ["абсурд", "повседневность"],\
    "mechanism" : ["обман ожидания"],\
    "actors": ["хозяйки"],\
}\
Тебе нужно обработать этот текст:\
'


def normalize_text(text: str) -> str:
    """
    Универсальная нормализация текста для сравнения и маппинга.
    Используется ВЕЗДЕ — и для actors, и для mechanism, и для theme.
    """
    if not isinstance(text, str):
        return ""
    
    text = text.lower().strip()
    
    # 1. Заменяем 'ё' на 'е' (критично для русского языка!)
    text = text.replace('ё', 'е')
    
    # 2. Убираем подчёркивания и дефисы (заменяем на пробелы)
    text = re.sub(r'[_-]', ' ', text)
    
    # 3. Убираем лишние пробелы
    text = re.sub(r'\s+', ' ', text).strip()
    
    # 4. Убираем пунктуацию в начале/конце
    text = re.sub(r'^[^\wа-я]+|[^\wа-я]+$', '', text, flags=re.UNICODE)
    
    # 5. Убираем "(ы)" и "(и)" — "программист(ы)" → "программист"
    text = re.sub(r'\([ыи]\)', '', text)
    
    # 6. Нормализация Unicode
    text = unicodedata.normalize('NFKC', text)
    
    return text.strip()


# =============================================================================
# 2. АДАПТИВНЫЙ ПОРОГ ЛЕВЕНШТЕЙНА
# =============================================================================

def adaptive_levenshtein_threshold(word: str, base_threshold: int = 2) -> int:
    """
    Адаптивный порог Левенштейна в зависимости от длины слова.
    Для коротких слов порог меньше (чтобы не было 'якуты' → 'кот').
    """
    length = len(word)
    if length <= 4:
        return 1  # Только 1 ошибка для коротких слов
    elif length <= 8:
        return base_threshold
    elif length <= 15:
        return base_threshold + 1
    else:
        return base_threshold + 2


# =============================================================================
# 3. УНИВЕРСАЛЬНАЯ ФУНКЦИЯ МАППИНГА
# =============================================================================

def build_mapping_to_canonical(
    observed_classes: Counter,
    canonical_classes: List[str],
    base_levenshtein_threshold: int = 2,
    verbose: bool = True,
    col_name: str = "",
) -> Dict[str, Optional[str]]:
    """
    Строит маппинг "наблюдённый_класс" → "канонический_класс" через Левенштейна.
    
    Если класс не удалось замаппить — возвращает None (не "другое"!).
    """
    mapping = {}
    unmapped = []
    suspicious = []
    
    for cls, count in observed_classes.items():
        if cls in canonical_classes:
            # Точное совпадение после нормализации
            mapping[cls] = cls
        else:
            # Адаптивный порог
            threshold = adaptive_levenshtein_threshold(cls, base_levenshtein_threshold)
            
            result = process.extractOne(
                cls,
                canonical_classes,
                scorer=distance.Levenshtein.distance,
                score_cutoff=threshold,
            )
            
            if result:
                matched, score, _ = result
                mapping[cls] = matched
                
                # Подозрительные маппинги (короткие слова с максимальным расстоянием)
                if score >= threshold and len(cls) <= 5:
                    suspicious.append((cls, matched, score, count))
            else:
                mapping[cls] = None  # Не замапилось
                unmapped.append((cls, count))
    
    if verbose:
        print(f"\n📋 [{col_name}] Примеры маппинга:")
        shown = 0
        for original, canonical in mapping.items():
            if original != canonical and canonical is not None:
                print(f"   '{original}' → '{canonical}'")
                shown += 1
                if shown >= 10:
                    break
        
        if suspicious:
            print(f"\n⚠️  [{col_name}] Подозрительные маппинги:")
            for original, matched, dist, count in suspicious[:5]:
                print(f"   '{original}' → '{matched}' (расст. {dist}, частота {count})")
        
        if unmapped:
            print(f"\n❌ [{col_name}] Не замапилось {len(unmapped)} классов (будут пропущены):")
            for cls, count in sorted(unmapped, key=lambda x: -x[1])[:10]:
                print(f"   '{cls}' ({count} раз)")
    
    return mapping


# =============================================================================
# 4. ПОСТРОЕНИЕ СЛОВАРЕЙ И МАППИНГОВ
# =============================================================================

def build_vocab_and_mapping(
    df: pd.DataFrame,
    columns: List[str],
    predefined_classes: Dict[str, List[str]],
    base_levenshtein_threshold: int = 2,
    verbose: bool = True,
) -> Tuple[Dict[str, Dict[str, int]], Dict[str, Dict[str, Optional[str]]]]:
    """
    Строит словари и маппинги с проверкой дубликатов.
    """
    vocab = {}
    all_mappings = {}
    
    for col in columns:
        if col not in df.columns:
            print(f"⚠️ Колонка '{col}' не найдена в DataFrame")
            continue
        if col not in predefined_classes:
            print(f"⚠️ Для колонки '{col}' нет предопределённых классов")
            continue
        
        canonical_classes = list(predefined_classes[col])
        
        # ПРОВЕРКА И УДАЛЕНИЕ ДУБЛИКАТОВ
        seen = set()
        unique_canonical = []
        duplicates = []
        for cls in canonical_classes:
            if cls in seen:
                duplicates.append(cls)
            else:
                seen.add(cls)
                unique_canonical.append(cls)
        
        if duplicates and verbose:
            print(f"\n⚠️ [{col}] Найдено {len(duplicates)} дубликатов в предопределённых классах:")
            for dup in set(duplicates):
                print(f"   '{dup}' встречается {duplicates.count(dup)+1} раз")
        
        canonical_classes = unique_canonical
        
        # Собираем все значения и нормализуем
        all_classes = []
        for val in df[col]:
            if isinstance(val, list):
                all_classes.extend([normalize_text(c) for c in val if normalize_text(c)])
            elif isinstance(val, str) and val.strip():
                norm = normalize_text(val)
                if norm:
                    all_classes.append(norm)
        
        class_counts = Counter(all_classes)
        
        if verbose:
            print(f"\n🔍 [{col}] Маппинг {len(class_counts)} уникальных классов на {len(canonical_classes)} предопределённых...")
        
        # Строим маппинг
        col_mapping = build_mapping_to_canonical(
            class_counts,
            canonical_classes,
            base_levenshtein_threshold=base_levenshtein_threshold,
            verbose=verbose,
            col_name=col,
        )
        
        # Словарь: индекс назначается ПО ПОРЯДКУ уникального списка
        vocab[col] = {cls: idx for idx, cls in enumerate(canonical_classes)}
        
        # ПРОВЕРКА: индексы должны быть от 0 до len-1
        if vocab[col]:
            max_idx = max(vocab[col].values())
            min_idx = min(vocab[col].values())
            expected_max = len(canonical_classes) - 1
            
            if max_idx != expected_max or min_idx != 0:
                print(f"⚠️ [{col}] Неправильные индексы: [{min_idx}, {max_idx}], ожидается [0, {expected_max}]")
            
            if verbose:
                print(f"   ✅ Размер словаря: {len(vocab[col])}, индексы [0, {max_idx}]")
        
        all_mappings[col] = col_mapping
        
        if verbose:
            mapped_count = sum(1 for v in col_mapping.values() if v is not None)
            print(f"   ✅ Замапилось: {mapped_count}/{len(col_mapping)} классов")
    
    return vocab, all_mappings


# =============================================================================
# 5. СОЗДАНИЕ БИНАРНЫХ ВЕКТОРОВ
# =============================================================================

def create_multihot_vectors(
    df: pd.DataFrame,
    vocab: Dict[str, Dict[str, int]],
    mapping: Dict[str, Dict[str, Optional[str]]],
) -> pd.DataFrame:
    """
    Создаёт бинарные векторы с проверкой индексов.
    """
    result_df = df.copy()
    
    for col, class_to_idx in vocab.items():
        if col not in df.columns:
            print(f"⚠️ Колонка '{col}' не найдена, пропускаем")
            continue
        
        if not class_to_idx:
            continue
        
        # ВАЖНО: размер вектора = max(индексы) + 1, а не len(class_to_idx)
        max_idx = max(class_to_idx.values())
        num_classes = max_idx + 1
        
        # Проверка: если есть пропуски в индексах, предупреждаем
        actual_indices = set(class_to_idx.values())
        expected_indices = set(range(num_classes))
        if actual_indices != expected_indices:
            print(f"⚠️ [{col}] Пропуски в индексах: {expected_indices - actual_indices}")
        
        col_mapping = mapping.get(col, {})
        
        print(f"\n🔨 Векторизация '{col}': {len(class_to_idx)} классов, размер вектора {num_classes}")
        
        vectors = []
        skipped_count = 0
        error_count = 0
        
        for val in df[col]:
            vec = [0] * num_classes
            
            if isinstance(val, list):
                items = val
            elif isinstance(val, str) and val.strip():
                items = [val]
            else:
                items = []
            
            for item in items:
                normalized = normalize_text(item)
                if not normalized:
                    continue
                
                canonical = col_mapping.get(normalized)
                
                if canonical is None:
                    skipped_count += 1
                    continue
                
                if canonical not in class_to_idx:
                    continue
                
                idx = class_to_idx[canonical]
                
                # ДОПОЛНИТЕЛЬНАЯ ПРОВЕРКА
                if 0 <= idx < num_classes:
                    vec[idx] = 1
                else:
                    if error_count < 5:
                        print(f"   ❌ Ошибка: '{canonical}' имеет индекс {idx}, но размер вектора {num_classes}")
                    error_count += 1
            
            vectors.append(vec)
        
        result_df[f"{col}_vector"] = vectors
        
        # Статистика
        total_active = sum(sum(v) for v in vectors)
        avg_active = total_active / len(vectors) if vectors else 0
        zero_vectors = sum(1 for v in vectors if sum(v) == 0)
        print(f"   ✅ Среднее активных классов: {avg_active:.2f}")
        print(f"   📊 Пустых векторов: {zero_vectors}/{len(vectors)} ({zero_vectors/len(vectors)*100:.1f}%)")
        if skipped_count > 0:
            print(f"   ⚠️ Пропущено элементов: {skipped_count}")
        if error_count > 0:
            print(f"   ❌ Всего ошибок индексации: {error_count}")
    
    return result_df

# =============================================================================
# 6. ГЛАВНАЯ ФУНКЦИЯ ПОСТПРОЦЕССИНГА
# =============================================================================

def postprocess_annotations(
    df: pd.DataFrame,
    mechanism_classes: List[str],
    theme_classes: List[str],
    top_n_actors: int = 200,
    min_actor_frequency: int = 10,
    base_levenshtein_threshold: int = 2,
    verbose: bool = True,
) -> Tuple[pd.DataFrame, Dict[str, Dict[str, int]], Dict[str, Dict[str, Optional[str]]], List[str]]:
    """
    Полный пайплайн постобработки:
    1. Определяет топ-N персонажей по частоте
    2. Строит словари и маппинги для всех трёх колонок
    3. Создаёт бинарные векторы
    
    Returns:
        (df_vectorized, vocab, mapping, top_actors)
    """
    result_df = df.copy()
    
    # --- Шаг 1: Определяем топ-N персонажей ---
    if verbose:
        print("\n" + "="*70)
        print("📊 ШАГ 1: Определение топ-N персонажей")
        print("="*70)
    
    all_actors = []
    for val in result_df["actors"]:
        if isinstance(val, list):
            all_actors.extend([normalize_text(a) for a in val if normalize_text(a)])
        elif isinstance(val, str) and val.strip():
            norm = normalize_text(val)
            if norm:
                all_actors.append(norm)
    
    actor_counts = Counter(all_actors)
    filtered = {a: c for a, c in actor_counts.items() if c >= min_actor_frequency}
    sorted_actors = sorted(filtered.items(), key=lambda x: x[1], reverse=True)
    top_actors = [a for a, _ in sorted_actors[:top_n_actors]]
    
    if verbose:
        total_mentions = sum(c for _, c in sorted_actors[:top_n_actors])
        coverage = total_mentions / len(all_actors) * 100 if all_actors else 0
        print(f"   Всего уникальных персонажей: {len(actor_counts)}")
        print(f"   Топ-{len(top_actors)} покрывают {coverage:.1f}% упоминаний")
        print(f"   Топ-10: {[a for a, _ in sorted_actors[:10]]}")
    
    # --- Шаг 2: Строим словари и маппинги ---
    if verbose:
        print("\n" + "="*70)
        print("📊 ШАГ 2: Построение словарей и маппингов")
        print("="*70)
    
    predefined = {
        "mechanism": mechanism_classes,
        "theme": theme_classes,
        "actors": top_actors,
    }
    
    vocab, mapping = build_vocab_and_mapping(
        result_df,
        columns=["mechanism", "theme", "actors"],
        predefined_classes=predefined,
        base_levenshtein_threshold=base_levenshtein_threshold,
        verbose=verbose,
    )
    
    # --- Шаг 3: Создаём бинарные векторы ---
    if verbose:
        print("\n" + "="*70)
        print("📊 ШАГ 3: Создание бинарных векторов")
        print("="*70)
    
    result_df = create_multihot_vectors(result_df, vocab, mapping)
    
    # --- Шаг 4: Нормализуем колонку actors в самом DataFrame ---
    # (чтобы в итоговом файле были канонические имена, а не исходные)
    actors_mapping = mapping["actors"]
    
    def normalize_actors_row(val):
        if isinstance(val, list):
            result = []
            for a in val:
                norm = normalize_text(a)
                if norm and norm in actors_mapping and actors_mapping[norm] is not None:
                    canonical = actors_mapping[norm]
                    if canonical not in result:
                        result.append(canonical)
            return result
        return []
    
    result_df["actors"] = result_df["actors"].apply(normalize_actors_row)
    
    return result_df, vocab, mapping, top_actors


# =============================================================================
# 7. СОХРАНЕНИЕ РЕЗУЛЬТАТОВ (для воспроизводимости)
# =============================================================================

def save_postprocessing_results(
    vocab: Dict[str, Dict[str, int]],
    mapping: Dict[str, Dict[str, Optional[str]]],
    top_actors: List[str],
    output_dir: str,
):
    """Сохраняет словари и маппинги для воспроизводимости на новых данных."""
    from pathlib import Path
    output_path = Path(output_dir)
    output_path.mkdir(exist_ok=True)
    
    # Словари (класс → индекс)
    with open(output_path / "vocab.json", 'w', encoding='utf-8') as f:
        json.dump(vocab, f, ensure_ascii=False, indent=2)
    
    # Маппинги (нормализованный → канонический)
    # Заменяем None на специальный маркер для JSON
    mapping_serializable = {}
    for col, m in mapping.items():
        mapping_serializable[col] = {k: (v if v is not None else "__UNMAPPED__") for k, v in m.items()}
    
    with open(output_path / "mapping.json", 'w', encoding='utf-8') as f:
        json.dump(mapping_serializable, f, ensure_ascii=False, indent=2)
    
    # Топ-актёры
    with open(output_path / "top_actors.json", 'w', encoding='utf-8') as f:
        json.dump(top_actors, f, ensure_ascii=False, indent=2)
    
    print(f"\n💾 Результаты сохранены в: {output_path}/")

def process_dataset(
    dataset_df_path: str,
    model_name: str,
    batch_size: int = 256,
    chunk_size: int = 10000,
    text_column: str = "Text",
    use_prefix_caching: bool = True,
    use_guided_decoding: bool = True,
):
    dataset_path = Path(dataset_df_path)
    output_path = dataset_path.with_name(f"{dataset_path.stem}_w_annot{dataset_path.suffix}")
    chunks_dir = dataset_path.with_name(f"{dataset_path.stem}_chunks")
    chunks_dir.mkdir(exist_ok=True)
    
    print(f"📂 Загрузка датасета: {dataset_path}")
    df = pd.read_parquet(dataset_path)
    
    if text_column not in df.columns:
        raise ValueError(f"Колонка '{text_column}' не найдена. Доступные: {df.columns.tolist()}")
        
    total_records = len(df)
    print(f"📊 Всего записей: {total_records}")
    
    # # --- Проверка чекпоинтов (чанков) ---
    # existing_chunks = sorted(chunks_dir.glob("chunk_*.parquet"))
    # start_idx = 0
    # for c in existing_chunks:
    #     try:
    #         start_idx += len(pd.read_parquet(c))
    #     except Exception:
    #         pass
            
    # print(f"🔄 Найдено чанков: {len(existing_chunks)}. Продолжаем с записи {start_idx}")
    
    # # --- Загрузка модели ---
    # print(f"🤖 Загрузка модели: {model_name}")
    # quant_type = None
    # name_lower = model_name.lower()
    # if "awq" in name_lower: quant_type = "awq"
    # elif "gptq" in name_lower: quant_type = "gptq_marlin"
    # elif "fp8" in name_lower: quant_type = "fp8"
    
    # llm = LLM(
    #     model=model_name,
    #     dtype="float16", # Предпочтительнее для современных GPU, стабильнее с квантами
    #     quantization=quant_type,
    #     gpu_memory_utilization=0.92,
    #     max_num_seqs=batch_size,
    #     enforce_eager=False, # ВАЖНО: False включает Flash-Attention и CUDA Graphs!
    #     enable_prefix_caching=use_prefix_caching, # Кеширует KV-cache для длинного SYSTEM_PROMPT
    #     disable_log_stats=True,
    # )

    # guided_params = StructuredOutputsParams(json=JSON_SCHEMA)
    # sampling_params = SamplingParams(
    #     temperature=0.1,
    #     max_tokens=512,
    #     stop=["}\n\n", "\n\n\n"],
    #     structured_outputs=guided_params,
    # )
    
    # # --- Обработка чанками ---
    # num_chunks = (total_records - start_idx + chunk_size - 1) // chunk_size
    # print(f"🚀 Начинаем обработку. Чанков осталось: {num_chunks}, размер чанка: {chunk_size}")
    
    # stats = {"total": 0, "valid": 0, "empty": 0}
    # start_chunk_idx = len(existing_chunks)
    
    # for i in tqdm(range(num_chunks), desc="Chunks"):
    #     chunk_start = start_idx + i * chunk_size
    #     chunk_end = min(chunk_start + chunk_size, total_records)
        
    #     if chunk_start >= total_records:
    #         break
            
    #     chunk_texts = df[text_column].iloc[chunk_start:chunk_end].tolist()
        
    #     try:
    #         prompts = [SYSTEM_PROMPT + str(text) for text in chunk_texts]
    #         outputs = llm.generate(prompts, sampling_params, use_tqdm=True)
            
    #         annotations = []
    #         for output in outputs:
    #             gen_text = output.outputs[0].text
    #             if use_guided_decoding:
    #                 try: annotations.append(normalize(json.loads(gen_text)))
    #                 except: annotations.append(extract_json(gen_text))
    #             else:
    #                 annotations.append(extract_json(gen_text))
                    
    #     except Exception as e:
    #         print(f"\n❌ Ошибка батча (возможно OOM): {e}. Переходим на обработку по одному...")
    #         annotations = []
    #         for text in chunk_texts:
    #             try:
    #                 out = llm.generate([SYSTEM_PROMPT + str(text)], sampling_params, use_tqdm=True)[0]
    #                 gen_text = out.outputs[0].text
    #                 if use_guided_decoding:
    #                     try: annotations.append(normalize(json.loads(gen_text)))
    #                     except: annotations.append(extract_json(gen_text))
    #                 else:
    #                     annotations.append(extract_json(gen_text))
    #             except Exception as e2:
    #                 print(f"  ⚠️ Пропуск: {e2}")
    #                 annotations.append({"setup": "", "punchline": "", "theme": [], "mechanism": [], "actors": []})
                    
    #     # Сохраняем чанк в отдельный Parquet файл (мгновенно и безопасно)
    #     chunk_df = pd.DataFrame(annotations)
    #     chunk_path = chunks_dir / f"chunk_{start_chunk_idx + i:04d}.parquet"
    #     chunk_df.to_parquet(chunk_path, index=False)
        
    #     # Статистика
    #     for ann in annotations:
    #         stats["total"] += 1
    #         if ann["setup"] or ann["punchline"]:
    #             stats["valid"] += 1
    #         else:
    #             stats["empty"] += 1
                
    #     print(f"\n💾 Чанк сохранен ({chunk_path.name}). Обработано: {stats['total']}/{total_records - start_idx} | "
    #           f"Успешно: {stats['valid']} | Пусто: {stats['empty']}")
              
    #     gc.collect()
    #     torch.cuda.empty_cache()

    # --- Финальная склейка ---
    print("🔗 Финальная склейка результатов...")
    final_chunks = sorted(chunks_dir.glob("chunk_*.parquet"))
    if not final_chunks:
        print("⚠️ Нет данных для склейки!")
        return
        
    df_annotations = pd.concat([pd.read_parquet(c) for c in final_chunks], ignore_index=True)
    
    # Приводим типы списков (Parquet иногда читает их как numpy arrays)
    for col in ["theme", "mechanism", "actors"]:
        df_annotations[col] = df_annotations[col].apply(lambda x: list(x) if x is not None else [])
        
    df_final = pd.concat([df.reset_index(drop=True), df_annotations], axis=1)

    print("\n🔍 Проверка предопределённых списков на дубликаты:")

    for name, classes in [("MECHANISM_CLASSES", MECHANISM_CLASSES), 
                        ("THEME_CLASSES", THEME_CLASSES)]:
        seen = set()
        duplicates = []
        for cls in classes:
            if cls in seen:
                duplicates.append(cls)
            seen.add(cls)
        
        if duplicates:
            print(f"   ⚠️ {name}: {len(duplicates)} дубликатов:")
            for dup in set(duplicates):
                print(f"      '{dup}'")
        else:
            print(f"   ✅ {name}: дубликатов нет")

    df_final, vocab, mapping, top_actors = postprocess_annotations(
        df_final,
        mechanism_classes=MECHANISM_CLASSES,
        theme_classes=THEME_CLASSES,
        top_n_actors=200,
        min_actor_frequency=10,
        base_levenshtein_threshold=2,
        verbose=True,
    )

    # Сохраняем словари для воспроизводимости
    save_postprocessing_results(vocab, mapping, top_actors, chunks_dir / "postprocessing")

    df_final.to_parquet(output_path, index=False)

    print(f"\n✅ Готово! Результат сохранен в: {output_path}")

if __name__ == "__main__":
    CLI(process_dataset, as_positional=False)
