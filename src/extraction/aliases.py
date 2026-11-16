"""Spelling aliases added to the lexicon as extra surfaces (canonical spellings follow the taxonomy).

Brand variants and city aliases are ported from the synthetic generator's normalizer
(y2_ai_synthetic_generator/synthetic_generation_app/normalize.py); Latin manufacturer names are added for queries
written in English ("Audi", "BMW").
"""
BRANDS = {  # יד_שנייה/מותגים
    "apple": "אפל", "samsung": "סמסונג", "סמסנג": "סמסונג", "xiaomi": "שיאומי", "שאומי": "שיאומי",
    "google": "גוגל", "oneplus": "וואןפלוס", "וואנפלוס": "וואןפלוס", "nokia": "נוקיה", "dell": "דל",
    "lenovo": "לנובו", "hp": "HP", "אייצפי": "HP", "אייצ׳פי": "HP", "asus": "אסוס", "microsoft": "מיקרוסופט",
    "lg": "LG", "אלגי": "LG", "sony": "סוני", "toshiba": "טושיבה",
    "bugaboo": "Bugaboo", "בוגבו": "Bugaboo", "באגבו": "Bugaboo", "בגבו": "Bugaboo",
    "cybex": "Cybex", "סייבקס": "Cybex", "סיבקס": "Cybex",
    "uppababy": "UPPAbaby", "אפהבייבי": "UPPAbaby", "אפאבייבי": "UPPAbaby", "אפבייבי": "UPPAbaby",
    "yoyo": "YOYO", "יויו": "YOYO",
}
CAR_MAKERS = {  # רכב/יצרנים
    "toyota": "טויוטה", "hyundai": "יונדאי", "kia": "קיה", "mazda": "מאזדה", "mercedes": "מרצדס",
    "mercedes-benz": "מרצדס", "benz": "מרצדס", "bmw": "ב.מ.וו", "במוו": "ב.מ.וו", "audi": "אאודי",
    "chevrolet": "שברולט", "suzuki": "סוזוקי", "tesla": "טסלה", "byd": "BYD", "mg": "MG",
}
CITIES = {  # עיר (both domains)
    "מודיעין": "מודיעין-מכבים-רעות", "תל אביב": "תל אביב-יפו", "תא": "תל אביב-יפו", "פת": "פתח תקווה",
    "פתח תקוה": "פתח תקווה", "ראשלצ": "ראשון לציון", "בש": "באר שבע", "ראשון": "ראשון לציון",
    "קרית אונו": "קריית אונו", "קרית גת": "קריית גת",
}
ALIASES = {"יד_שנייה/מותגים": BRANDS, "רכב/יצרנים": CAR_MAKERS, "נדל״ן/עיר": CITIES, "יד_שנייה/עיר": CITIES}
