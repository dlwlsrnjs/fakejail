"""Shared PC2 language registry and JailNewsBench source-language mapping."""

from __future__ import annotations


LANGUAGES = {
    "Albanian": "als_Latn", "Amharic": "amh_Ethi", "Arabic": "arb_Arab",
    "Armenian": "hye_Armn", "Azerbaijani": "azj_Latn", "Bengali": "ben_Beng",
    "Bosnian": "bos_Latn", "Bulgarian": "bul_Cyrl", "Burmese": "mya_Mymr",
    "Cantonese": "yue_Hant", "Catalan": "cat_Latn", "Croatian": "hrv_Latn",
    "Czech": "ces_Latn", "Danish": "dan_Latn", "Dutch": "nld_Latn",
    "English": "eng_Latn", "Estonian": "est_Latn", "Filipino": "tgl_Latn",
    "Finnish": "fin_Latn", "French": "fra_Latn", "Georgian": "kat_Geor",
    "German": "deu_Latn", "Greek": "ell_Grek", "Haitian Creole": "hat_Latn",
    "Hebrew": "heb_Hebr", "Hindi": "hin_Deva", "Hungarian": "hun_Latn",
    "Icelandic": "isl_Latn", "Indonesian": "ind_Latn", "Irish": "gle_Latn",
    "Italian": "ita_Latn", "Japanese": "jpn_Jpan", "Kazakh": "kaz_Cyrl",
    "Khmer": "khm_Khmr", "Kinyarwanda": "kin_Latn", "Korean": "kor_Hang",
    "Kyrgyz": "kir_Cyrl", "Lao": "lao_Laoo", "Latvian": "lvs_Latn",
    "Lithuanian": "lit_Latn", "Luxembourgish": "ltz_Latn", "Malagasy": "plt_Latn",
    "Malay": "zsm_Latn", "Maltese": "mlt_Latn", "Mandarin Chinese": "zho_Hans",
    "Mongolian": "khk_Cyrl", "Montenegrin": "srp_Cyrl", "Nepali": "npi_Deva",
    "Norwegian": "nob_Latn", "Pashto": "pbt_Arab", "Persian": "pes_Arab",
    "Polish": "pol_Latn", "Portuguese": "por_Latn", "Romanian": "ron_Latn",
    "Russian": "rus_Cyrl", "Serbian": "srp_Cyrl", "Shona": "sna_Latn",
    "Sinhala": "sin_Sinh", "Slovak": "slk_Latn", "Slovene": "slv_Latn",
    "Spanish": "spa_Latn", "Swahili": "swh_Latn", "Swedish": "swe_Latn",
    "Tajik": "tgk_Cyrl", "Thai": "tha_Thai", "Turkish": "tur_Latn",
    "Turkmen": "tuk_Latn", "Ukrainian": "ukr_Cyrl", "Urdu": "urd_Arab",
    "Uzbek": "uzn_Latn", "Vietnamese": "vie_Latn", "Zulu": "zul_Latn",
}


SOURCE_TO_NLLB = {
    "bg": "bul_Cyrl", "cs": "ces_Latn", "de": "deu_Latn", "el": "ell_Grek",
    "en": "eng_Latn", "es": "spa_Latn", "fr": "fra_Latn", "hu": "hun_Latn",
    "id": "ind_Latn", "it": "ita_Latn", "ja": "jpn_Jpan", "ko": "kor_Hang",
    "lt": "lit_Latn", "lv": "lvs_Latn", "nl": "nld_Latn", "no": "nob_Latn",
    "pl": "pol_Latn", "pt": "por_Latn", "ro": "ron_Latn", "sk": "slk_Latn",
    "sl": "slv_Latn", "sv": "swe_Latn", "zh": "zho_Hans",
}


# Language names accepted by Xiaomi's translation-specialized MiLMMT-46 model.
MILMMT_LANGUAGE_NAMES = {
    "Arabic": "Arabic", "Azerbaijani": "Azerbaijani", "Bulgarian": "Bulgarian",
    "Bengali": "Bengali", "Catalan": "Catalan", "Czech": "Czech",
    "Danish": "Danish", "German": "German", "Greek": "Greek",
    "English": "English", "Spanish": "Spanish", "Persian": "Persian",
    "Finnish": "Finnish", "French": "French", "Hebrew": "Hebrew",
    "Hindi": "Hindi", "Croatian": "Croatian", "Hungarian": "Hungarian",
    "Indonesian": "Indonesian", "Italian": "Italian", "Japanese": "Japanese",
    "Kazakh": "Kazakh", "Khmer": "Khmer", "Korean": "Korean", "Lao": "Lao",
    "Malay": "Malay", "Burmese": "Burmese", "Norwegian": "Norwegian",
    "Dutch": "Dutch", "Polish": "Polish", "Portuguese": "Portuguese",
    "Romanian": "Romanian", "Russian": "Russian", "Slovak": "Slovak",
    "Slovene": "Slovenian", "Swedish": "Swedish", "Thai": "Thai",
    "Filipino": "Tagalog", "Turkish": "Turkish", "Urdu": "Urdu",
    "Uzbek": "Uzbek", "Vietnamese": "Vietnamese", "Cantonese": "Cantonese",
    "Mandarin Chinese": "Chinese (Simplified)",
}


# MADLAD-400 target tags. This route covers the PC2 languages not supported by
# MiLMMT-46 and remains available as a genuinely different MT family.
MADLAD_CODES = {
    "Albanian": "sq", "Amharic": "am", "Arabic": "ar", "Armenian": "hy",
    "Azerbaijani": "az", "Bengali": "bn", "Bosnian": "bs", "Bulgarian": "bg",
    "Burmese": "my", "Cantonese": "zh_Hant", "Catalan": "ca", "Croatian": "hr",
    "Czech": "cs", "Danish": "da", "Dutch": "nl", "English": "en",
    "Estonian": "et", "Filipino": "fil", "Finnish": "fi", "French": "fr",
    "Georgian": "ka", "German": "de", "Greek": "el", "Haitian Creole": "ht",
    "Hebrew": "he", "Hindi": "hi", "Hungarian": "hu", "Icelandic": "is",
    "Indonesian": "id", "Irish": "ga", "Italian": "it", "Japanese": "ja",
    "Kazakh": "kk", "Khmer": "km", "Kinyarwanda": "rw", "Korean": "ko",
    "Kyrgyz": "ky", "Lao": "lo", "Latvian": "lv", "Lithuanian": "lt",
    "Luxembourgish": "lb", "Malagasy": "mg", "Malay": "ms", "Maltese": "mt",
    "Mandarin Chinese": "zh", "Mongolian": "mn", "Montenegrin": "sr",
    "Nepali": "ne", "Norwegian": "no", "Pashto": "ps", "Persian": "fa",
    "Polish": "pl", "Portuguese": "pt", "Romanian": "ro", "Russian": "ru",
    "Serbian": "sr", "Shona": "sn", "Sinhala": "si", "Slovak": "sk",
    "Slovene": "sl", "Spanish": "es", "Swahili": "sw", "Swedish": "sv",
    "Tajik": "tg", "Thai": "th", "Turkish": "tr", "Turkmen": "tk",
    "Ukrainian": "uk", "Urdu": "ur", "Uzbek": "uz", "Vietnamese": "vi",
    "Zulu": "zu",
}
