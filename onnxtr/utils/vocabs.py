# Copyright (C) 2021-2024, Mindee | Felix Dittrich.

# This program is licensed under the Apache License 2.0.
# See LICENSE or go to <https://opensource.org/licenses/Apache-2.0> for full license details.

import string
from typing import Dict

__all__ = ["VOCABS"]


VOCABS: Dict[str, str] = {
    "digits": string.digits,
    "ascii_letters": string.ascii_letters,
    "punctuation": string.punctuation,
    "currency": "£€¥¢฿",
    "ancient_greek": "αβγδεζηθικλμνξοπρστυφχψωΑΒΓΔΕΖΗΘΙΚΛΜΝΞΟΠΡΣΤΥΦΧΨΩ",
    "arabic_letters": "ءآأؤإئابةتثجحخدذرزسشصضطظعغـفقكلمنهوىي",
    "persian_letters": "پچڢڤگ",
    "hindi_digits": "٠١٢٣٤٥٦٧٨٩",
    "arabic_diacritics": "ًٌٍَُِّْ",
    "arabic_punctuation": "؟؛«»—",
}

VOCABS["latin"] = VOCABS["digits"] + VOCABS["ascii_letters"] + VOCABS["punctuation"]
VOCABS["english"] = VOCABS["latin"] + "°" + VOCABS["currency"]
VOCABS["legacy_french"] = VOCABS["latin"] + "°" + "àâéèêëîïôùûçÀÂÉÈËÎÏÔÙÛÇ" + VOCABS["currency"]
VOCABS["french"] = VOCABS["english"] + "àâéèêëîïôùûüçÀÂÉÈÊËÎÏÔÙÛÜÇ"
VOCABS["portuguese"] = VOCABS["english"] + "áàâãéêíïóôõúüçÁÀÂÃÉÊÍÏÓÔÕÚÜÇ"
VOCABS["spanish"] = VOCABS["english"] + "áéíóúüñÁÉÍÓÚÜÑ" + "¡¿"
VOCABS["italian"] = VOCABS["english"] + "àèéìíîòóùúÀÈÉÌÍÎÒÓÙÚ"
VOCABS["german"] = VOCABS["english"] + "äöüßÄÖÜẞ"
VOCABS["arabic"] = (
    VOCABS["digits"]
    + VOCABS["hindi_digits"]
    + VOCABS["arabic_letters"]
    + VOCABS["persian_letters"]
    + VOCABS["arabic_diacritics"]
    + VOCABS["arabic_punctuation"]
    + VOCABS["punctuation"]
)
VOCABS["czech"] = VOCABS["english"] + "áčďéěíňóřšťúůýžÁČĎÉĚÍŇÓŘŠŤÚŮÝŽ"
VOCABS["polish"] = VOCABS["english"] + "ąćęłńóśźżĄĆĘŁŃÓŚŹŻ"
VOCABS["dutch"] = VOCABS["english"] + "áéíóúüñÁÉÍÓÚÜÑ"
VOCABS["norwegian"] = VOCABS["english"] + "æøåÆØÅ"
VOCABS["danish"] = VOCABS["english"] + "æøåÆØÅ"
VOCABS["finnish"] = VOCABS["english"] + "äöÄÖ"
VOCABS["swedish"] = VOCABS["english"] + "åäöÅÄÖ"
VOCABS["vietnamese"] = (
    VOCABS["english"]
    + "áàảạãăắằẳẵặâấầẩẫậéèẻẽẹêếềểễệóòỏõọôốồổộỗơớờởợỡúùủũụưứừửữựiíìỉĩịýỳỷỹỵ"
    + "ÁÀẢẠÃĂẮẰẲẴẶÂẤẦẨẪẬÉÈẺẼẸÊẾỀỂỄỆÓÒỎÕỌÔỐỒỔỘỖƠỚỜỞỢỠÚÙỦŨỤƯỨỪỬỮỰIÍÌỈĨỊÝỲỶỸỴ"
)
VOCABS["hebrew"] = VOCABS["english"] + "אבגדהוזחטיכלמנסעפצקרשת" + "₪"
VOCABS["multilingual"] = "".join(
    dict.fromkeys(
        VOCABS["french"]
        + VOCABS["portuguese"]
        + VOCABS["spanish"]
        + VOCABS["german"]
        + VOCABS["czech"]
        + VOCABS["polish"]
        + VOCABS["dutch"]
        + VOCABS["italian"]
        + VOCABS["norwegian"]
        + VOCABS["danish"]
        + VOCABS["finnish"]
        + VOCABS["swedish"]
        + "§"
    )
)
