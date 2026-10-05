"""
preprocess.py (VARIAN EKSPERIMEN: SELURUH EMOJI DIHAPUS): praproses teks ulasan untuk
fine-tuning IndoBERT (Subbab 3.3).

Perbedaan satu-satunya terhadap varian utama: tidak ada kamus emoji. Seluruh emoji dan simbol
pictograf dihapus (diganti spasi). Langkah lain (case folding, URL dan username, elongasi,
normalisasi slang, duplikat murni) identik dengan varian utama.

Praproses dijaga minimal (tanpa stemming dan stopword removal). Filter bahasa sudah
dilakukan saat scraping (kolom detected_lang) dan tidak diulang di sini.

Berkas kamus slang (lexicon) adalah masukan yang dipertahankan di Drive. Modul ini
TIDAK mengunduh kamus secara otomatis, supaya run murni tidak bergantung pada isi
sumber daring yang dapat berubah. Hash kamus dicatat pada log lingkungan.
"""

import os
import re

import pandas as pd

from src import config, drive_io

# ----------------------------------------------------------------------
# Penghapus seluruh emoji dan simbol pictograf (diganti spasi)
# ----------------------------------------------------------------------
_EMOJI_PATTERN = re.compile(
    "["
    "\U0001F000-\U0001FAFF"   # mahjong, kartu, huruf berlingkar, emoji, bendera, piktograf
    "\U00002600-\U000027BF"   # simbol lain dan dingbat (hati, bintang, centang, dsb.)
    "\U00002190-\U000021FF"   # panah
    "\U00002300-\U000023FF"   # simbol teknis (jam, timer, dsb.)
    "\U00002934\U00002935"    # panah lengkung
    "\U00002B00-\U00002BFF"   # simbol dan panah lain (termasuk bintang U+2B50)
    "\U0000203C\U00002049"    # tanda seru ganda dan tanda tanya-seru
    "\U000020E3"              # keycap penggabung
    "\U00003030\U0000303D\U00003297\U00003299"
    "\U000E0020-\U000E007F"   # tag karakter (bendera subdivisi)
    "\U0000FE0F\U0000200D"    # variation selector dan zero-width joiner
    "]+",
    flags=re.UNICODE,
)

REQUIRED_SLANG_COLUMNS = ["slang", "formal"]
KAMUS_ALAY_RUJUKAN = "https://github.com/nasalsabila/kamus-alay (colloquial-indonesian-lexicon.csv)"


def _validate_slang_columns(df, source_path):
    """Validasi eksplisit kolom kamus (Subbab 3.3, butir 5)."""
    missing = [c for c in REQUIRED_SLANG_COLUMNS if c not in df.columns]
    if missing:
        raise ValueError(
            f"Kamus slang di '{source_path}' tidak memiliki kolom yang diharapkan: {missing}. "
            f"Kolom tersedia: {df.columns.tolist()}. Kemungkinan format sumber kamus berubah "
            f"(rujukan: {KAMUS_ALAY_RUJUKAN})."
        )


def load_slang_lexicon(base_path=None, domain_path=None):
    """
    Memuat dan menggabungkan dua kamus: slang_base.csv (Kamus Alay, wajib) dan
    slang_domain.csv (entri khusus domain, opsional; bila tidak ada, dilewati).
    """
    base_path = base_path or config.SLANG_BASE_FILE
    domain_path = domain_path or config.SLANG_DOMAIN_FILE
    if not os.path.exists(base_path):
        raise FileNotFoundError(
            f"[BERHENTI] Kamus dasar tidak ditemukan: {base_path}. Folder lexicon adalah masukan "
            f"yang dipertahankan; salin kembali berkasnya ke Drive."
        )
    lexicon = {}
    df_base = pd.read_csv(base_path)
    _validate_slang_columns(df_base, base_path)
    df_base = df_base[REQUIRED_SLANG_COLUMNS]
    lexicon.update(dict(zip(df_base["slang"], df_base["formal"])))
    n_base = len(df_base)

    n_domain_new = 0
    if os.path.exists(domain_path):
        df_domain = pd.read_csv(domain_path)
        _validate_slang_columns(df_domain, domain_path)
        before = len(lexicon)
        lexicon.update(dict(zip(df_domain["slang"], df_domain["formal"])))
        n_domain_new = len(lexicon) - before
        print(f"[OK] Kamus domain dimuat: {len(df_domain)} entri ({n_domain_new} baru).")
    else:
        print(f"[INFO] Kamus domain tidak ditemukan ({domain_path}); hanya Kamus Alay yang dipakai.")
    print(f"[OK] Kamus dasar dimuat: {n_base} entri.")
    return lexicon


_SLANG_DICT = None


def get_slang_dict():
    global _SLANG_DICT
    if _SLANG_DICT is None:
        _SLANG_DICT = load_slang_lexicon()
    return _SLANG_DICT


# ----------------------------------------------------------------------
# Fungsi pembersihan
# ----------------------------------------------------------------------
def strip_all_emoji(text):
    """Menghapus seluruh emoji dan simbol pictograf (diganti spasi)."""
    return _EMOJI_PATTERN.sub(" ", text)


_WORD_STRIP_PATTERN = re.compile(r"^(\W*)(\w+)(\W*)$", flags=re.UNICODE)


def _normalize_word_with_slang(word, slang_dict):
    """Tanda baca di tepi kata dipisahkan sementara sebelum pencarian kamus (Subbab 3.3, butir 4)."""
    match = _WORD_STRIP_PATTERN.match(word)
    if not match:
        return word
    prefix, core, suffix = match.groups()
    return f"{prefix}{slang_dict.get(core, core)}{suffix}"


def clean_text_for_bert(text, slang_dict=None):
    if not isinstance(text, str):
        return ""
    slang_dict = get_slang_dict() if slang_dict is None else slang_dict

    text = text.lower()
    text = re.sub(r"http\S+|www\S+|https\S+", "", text, flags=re.MULTILINE)
    text = re.sub(r"@\w+", "", text)

    text = strip_all_emoji(text)

    text = re.sub(r"(.)\1{2,}", r"\1\1", text)
    text = re.sub(r"([!?.,])\1+", r"\1", text)

    if slang_dict:
        words = text.split()
        text = " ".join(_normalize_word_with_slang(w, slang_dict) for w in words)

    text = re.sub(r"\s+", " ", text).strip()
    return text


# ----------------------------------------------------------------------
# Analitik dan pipeline
# ----------------------------------------------------------------------
def compute_slang_coverage(df, slang_dict, raw_col="review_text"):
    """Estimasi kasar cakupan kamus dari teks mentah (bukan angka final pipeline)."""
    all_words = " ".join(df[raw_col].dropna().astype(str).str.lower()).split()
    total = len(all_words)
    cleaned_words = [re.sub(r"[^a-z0-9]", "", w) for w in all_words]
    matched = sum(1 for w in cleaned_words if w in slang_dict)
    return {"total_words": total, "normalized_words": matched,
            "coverage_pct": round(matched / total * 100, 2) if total > 0 else 0.0}


def drop_text_rating_conflicts(df, text_col="cleaned_text", verbose=True):
    """
    Resolusi teks identik dengan rating berbeda melalui voting mayoritas (Subbab 3.3, butir 8):
    baris dengan rating minoritas dibuang, grup tanpa mayoritas jelas (seri) dibuang seluruhnya.

    CATATAN: karena duplikat murni (teks dan rating identik) sudah dihapus pada run_preprocessing,
    setiap grup konflik memuat tepat satu baris per rating. Semua grup konflik karena itu seri,
    dan seluruhnya dibuang. Jumlah baris minoritas yang dilaporkan akan selalu nol.
    """
    dup_mask = df.duplicated(subset=[text_col], keep=False)
    df_dup = df[dup_mask]
    df_unique = df[~dup_mask]

    keep_groups = []
    n_dropped_minority = 0
    n_dropped_tie = 0

    for _, group in df_dup.groupby(text_col):
        rating_counts = group["rating"].value_counts()
        if len(rating_counts) == 1:
            keep_groups.append(group)
            continue
        top_count = rating_counts.iloc[0]
        modes = rating_counts[rating_counts == top_count].index.tolist()
        if len(modes) > 1:
            n_dropped_tie += len(group)
            continue
        majority_rating = modes[0]
        kept = group[group["rating"] == majority_rating]
        n_dropped_minority += len(group) - len(kept)
        keep_groups.append(kept)

    df_result = pd.concat([df_unique] + keep_groups, ignore_index=True) if keep_groups else df_unique
    if verbose:
        print(f"Baris dibuang (rating minoritas dalam grup konflik): {n_dropped_minority}")
        print(f"Baris dibuang (seri, tidak ada mayoritas jelas)    : {n_dropped_tie}")
        print(f"Total dibuang                                      : {n_dropped_minority + n_dropped_tie}")
    return df_result


def compute_token_truncation_rate(df, text_col="cleaned_text", max_len=None, tokenizer=None):
    from transformers import AutoTokenizer
    max_len = max_len or config.MAX_LEN
    tokenizer = tokenizer or AutoTokenizer.from_pretrained(config.PRETRAINED_MODEL_NAME)
    token_lengths = df[text_col].astype(str).apply(
        lambda t: len(tokenizer.encode(t, add_special_tokens=True)))
    pct_truncated = (token_lengths > max_len).mean() * 100
    return {"max_len": max_len, "pct_truncated": round(float(pct_truncated), 2),
            "median_token_len": int(token_lengths.median())}


def run_preprocessing(master_path=None, tokenizer=None, write=True):
    master_path = master_path or config.MASTER_FILE
    slang_dict = get_slang_dict()

    df = pd.read_csv(master_path)
    initial_len = len(df)
    print(f"Jumlah data awal: {initial_len} baris")

    df = df.dropna(subset=["review_text"])
    n_with_emoji = int(df["review_text"].astype(str).apply(
        lambda t: bool(_EMOJI_PATTERN.search(t))).sum())
    print(f"Ulasan yang memuat emoji/simbol (dihapus seluruhnya): {n_with_emoji}")

    df["cleaned_text"] = df["review_text"].apply(lambda t: clean_text_for_bert(t, slang_dict))

    slang_coverage = compute_slang_coverage(df, slang_dict)
    print(f"Cakupan kamus slang (estimasi dari teks mentah): "
          f"{slang_coverage['normalized_words']}/{slang_coverage['total_words']} kata "
          f"({slang_coverage['coverage_pct']}%)")

    n_empty = int((df["cleaned_text"].str.strip() == "").sum())
    print(f"Ulasan kosong setelah pembersihan (dibuang): {n_empty}")
    df = df[df["cleaned_text"].str.strip() != ""]

    word_counts = df["cleaned_text"].astype(str).str.split().apply(len)
    is_very_short = word_counts <= 3
    df["cleaned_text_word_count"] = word_counts
    df["is_very_short_text"] = is_very_short
    print(f"Teks sangat pendek (tiga kata atau kurang): {is_very_short.mean() * 100:.2f}%")

    dup_mask = df.duplicated(subset=["cleaned_text"], keep=False)
    conflicting = df[dup_mask].groupby("cleaned_text")["rating"].nunique()
    n_conflicting = int((conflicting > 1).sum())
    print(f"Teks identik dengan rating berbeda (setelah pembersihan): {n_conflicting} grup")
    print("Resolusi voting mayoritas dijalankan setelah deteksi Confident learning (Subbab 3.3, 3.6).")

    df = df.drop_duplicates(subset=["cleaned_text", "rating"])  # hanya duplikat murni
    final_len = len(df)
    print(f"Setelah dibersihkan: {final_len} baris (terbuang: {initial_len - final_len})")

    truncation = compute_token_truncation_rate(df, tokenizer=tokenizer) if tokenizer is not False else {}
    if truncation:
        print(f"Ulasan yang terpotong (> {truncation['max_len']} token): {truncation['pct_truncated']}%")

    summary = {
        "initial_rows": initial_len, "final_rows": final_len,
        "rows_dropped": initial_len - final_len,
        "emoji_mode": "hapus_semua",
        "n_ulasan_beremoji": n_with_emoji,
        "n_kosong_setelah_pembersihan": n_empty,
        "text_rating_conflict_groups": n_conflicting,
        "pct_very_short_text": round(float(is_very_short.mean() * 100), 2),
        **{f"truncation_{k}": v for k, v in truncation.items()},
        **{f"slang_{k}": v for k, v in slang_coverage.items()},
    }
    if write:
        drive_io.write_once_csv(df, config.CLEAN_TEXT_FILE)
        drive_io.write_once_csv(pd.DataFrame([summary]),
                                os.path.join(config.SHARED_RESULTS_DIR, "preprocessing_summary.csv"))
    return df, summary