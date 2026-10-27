"""Multilingual keyword / negation rules - a cheap cross-check for the LLM extraction.

Output per label: "positive" | "negative" | "unknown". Deliberately conservative: it flags
sentences that mention the structure together with a pathology term and no negation cue.
Use it to (a) audit LLM disagreements, (b) fall back when an LLM call failed.
"""
from __future__ import annotations

import re
from dataclasses import dataclass

from ..constants import LABELS

_SENT_SPLIT = re.compile(r"(?<=[.;!?。])\s+|\n+")

NEGATION = [
    # en
    r"\bno\b", r"\bnot\b", r"\bwithout\b", r"\bintact\b", r"\bnormal\b", r"\bunremarkable\b", r"\babsen[ct]e?\b",
    r"\bnegative for\b", r"\bpreserved\b",
    # es / pt / it
    r"\bsin\b", r"\bsem\b", r"\bsenza\b", r"\bausencia\b", r"\bausência\b", r"\bassenza\b", r"\bíntegr[oa]s?\b",
    r"\bintegr[oa]s?\b", r"\bíntegra\b", r"\bno se (?:observa|evidencia|identifica)\b", r"\bnão\b", r"\bnon\b",
    r"\bnella norma\b", r"\bregolar[ei]\b", r"\bconservad[oa]s?\b",
    # fr
    r"\bpas d[e']\b", r"\bsans\b", r"\baucun[e]?\b", r"\bintacts?\b", r"\bnormale?s?\b",
    # de / nl
    r"\bkein[esr]?\b", r"\bohne\b", r"\bintakt\b", r"\bregelrecht\b", r"\bunauffällig\b", r"\bnicht\b",
    r"\bgeen\b", r"\bzonder\b", r"\bnormaal\b",
    # pl / tr
    r"\bbez\b", r"\bbrak\b", r"\bnie\b", r"\bprawidłow", r"\byok\b", r"\bizlenmedi\b", r"\bsaptanmadı\b",
    r"\bgörülmedi\b", r"\bintakt\b", r"\bdoğal\b",
    # ru / uk
    r"\bнет\b", r"\bбез\b", r"\bне\b", r"отсутств", r"интактн", r"не измен", r"не выявлен", r"не определя",
    r"сохранен",
    # hr / sr / bs
    r"\bbez\b", r"\buredn", r"\bodržan", r"\bprimjeren", r"\bne vidi se\b", r"\bnema\b", r"\bočuvan",
    # el
    r"\bχωρίς\b", r"\bδεν\b", r"φυσιολογικ", r"\bάνευ\b",
    # bg
    r"\bне се\b", r"\bнормал", r"\bлипс",
    # zh / ja / ko
    r"未见", r"无", r"没有", r"正常", r"完整", r"なし", r"認めない", r"認めず", r"正常", r"없음", r"정상",
]
_NEG_RE = re.compile("|".join(NEGATION), re.IGNORECASE)

HEDGE = [r"\bpossible\b", r"\bcannot exclude\b", r"\bsuspect", r"\bprobable\b", r"\bquestionable\b", r"\bversus\b",
         r"\bvs\.?\b", r"\bposible\b", r"\bsospech", r"\bsuspicion\b", r"\bmöglich", r"\bverdacht\b", r"\bfraglich",
         r"возможн", r"подозр", r"нельзя исключить", r"\?"]
_HEDGE_RE = re.compile("|".join(HEDGE), re.IGNORECASE)

TEAR = [r"\btears?\b", r"\btorn\b", r"\bruptur", r"\brotura\b", r"\bdesgarro\b", r"\bfissur", r"\blésion", r"\blesion",
        r"\brottura\b", r"\briss\b", r"\bein-?riss", r"\bläsion", r"разрыв", r"поврежден", r"\bscheur", r"\buszkodz",
        r"\byırtı", r"撕裂", r"断裂", r"파열", r"\bdisrupt", r"\bdiscontinu", r"\bmeniscocapsular", r"\bramp\b",
        r"\bbucket", r"\bflap\b", r"\bradial\b", r"\bhorizontal\b", r"\broot\b", r"\bcomplex\b", r"\bsprain",
        r"\besguince\b", r"\bentorse\b", r"\bdistorsion", r"\bzerrung", r"\bgrade\s*[1-3I]{1,3}\b", r"\bgrad\s*[1-3I]{1,3}\b",
        r"\boedema\b", r"\bedema\b", r"\bödem", r"отек", r"\bpartial\b", r"\bcomplete\b", r"\bruptur",
        r"\byırtı[kğ]", r"\brüptür", r"\bkopma", r"разкъсван", r"ρήξ", r"\bprekid", r"\blezij", r"\bdegenerat", r"\bdejeneras",
        r"\bgrade\s*(?:ii|iii|2|3)\b", r"\bgrad\s*(?:ii|iii|2|3)\b", r"\bstupnja\b", r"\bmiksoid", r"\bmukoid", r"\bmucoid"]
_TEAR_RE = re.compile("|".join(TEAR), re.IGNORECASE)

OA_TERM = [r"osteoarthr", r"arthros", r"artros", r"artrosi", r"arthrose", r"gonarthro", r"chondr", r"condr", r"cartila",
           r"knorpel", r"osteophyt", r"osteofit", r"ostéophyt", r"хрящ", r"остеофит", r"хондромаляц", r"артроз",
           r"kraakbeen", r"chrząstk", r"kıkırdak", r"kondromalazi", r"软骨", r"骨关节炎", r"骨赘", r"연골", r"degenerat",
           r"joint space", r"subchondral", r"subcondral", r"sous-chondral", r"субхондральн", r"hrskavic", r"hondromalacij",
           r"χόνδρ", r"οστεοφ", r"хрущял", r"остеофит", r"kıkırdak", r"kondromalazi", r"osteofit", r"gonartro", r"rotulian",
           r"patelar", r"kraakbeen", r"osteofyt", r"chondropath", r"chondrose", r"chondrosis", r"kartilaj"]
_OA_RE = re.compile("|".join(OA_TERM), re.IGNORECASE)

STRUCT: dict[str, list[str]] = {
    "acl": [r"\bacl\b", r"anterior cruciate", r"\blca\b", r"cruzado anterior", r"croisé antérieur", r"crociato anteriore",
            r"vorder\w* kreuzband", r"\bvkb\b", r"передн\w+ крестообразн", r"\bпкс\b", r"voorste kruisband",
            r"krzyżowe przednie", r"ön çapraz", r"前十字", r"前交叉", r"전방십자", r"\bacl graft\b", r"prednj\w* križn",
            r"\bpkl\b", r"πρόσθι\w* χιαστ", r"\bπχσ\b", r"предна кръстна", r"anterior çapraz", r"\böçb\b"],
    "mcl": [r"\bmcl\b", r"medial collateral", r"colateral medial", r"colateral interno", r"collatéral (?:médial|interne)",
            r"\blcm\b", r"\blli\b", r"collaterale mediale", r"innenband", r"mediales seitenband", r"\bmsb\b",
            r"(?:внутренн|медиальн)\w+ коллатеральн", r"\bмкс\b", r"mediale collaterale band", r"poboczne piszczelowe",
            r"iç yan bağ", r"内侧副韧带", r"내측측부인대", r"tibial collateral", r"medijaln\w* kolateraln", r"unutarnj\w* kolateraln",
            r"έσω πλάγι", r"медиал\w* колатерал", r"vътрешн\w* колатерал", r"medial kollateral", r"medial kolateral", r"\bmkl\b"],
    "medial_meniscus": [r"medial meniscus", r"medial meniscal", r"menisco (?:medial|interno)", r"ménisque (?:interne|médial)",
                        r"menisco (?:mediale|interno)", r"innenmeniskus", r"(?:медиальн|внутренн)\w+ мениск",
                        r"mediale meniscus", r"łąkotk\w+ przyśrodkow", r"iç menisküs", r"内侧半月板", r"내측 반월",
                        r"medijaln\w* menisk", r"unutarnj\w* menisk", r"έσω μηνίσκ", r"медиалн\w* мениск", r"вътрешн\w* мениск",
                        r"medial menisküs", r"medyal menisküs", r"binnenmeniscus"],
    "lateral_meniscus": [r"lateral meniscus", r"lateral meniscal", r"menisco (?:lateral|externo)", r"ménisque (?:externe|latéral)",
                         r"menisco (?:laterale|esterno)", r"außenmeniskus", r"aussenmeniskus", r"(?:латеральн|наружн)\w+ мениск",
                         r"laterale meniscus", r"łąkotk\w+ boczn", r"dış menisküs", r"外侧半月板", r"외측 반월",
                         r"lateraln\w* menisk", r"vanjsk\w* menisk", r"έξω μηνίσκ", r"латералн\w* мениск", r"външн\w* мениск",
                         r"lateral menisküs", r"buitenmeniscus"],
    "medial_oa": [r"medial (?:compartment|tibiofemoral|femorotibial|femoral condyle|tibial plateau)", r"compartimento medial",
                  r"compartimento interno", r"compartiment (?:interne|fémoro-tibial interne|médial)", r"medial\w* kompartiment",
                  r"медиальн\w+ (?:отдел|мыщел)", r"내측 구획", r"内侧间室", r"medial femoral", r"medial tibial",
                  r"medijaln\w* (?:kompartment|kondil|tibijaln)", r"έσω (?:διαμέρισμ|μηριαί|κνημιαί)", r"medial (?:kompartman|femoral|tibial)",
                  r"mediaal", r"medijalnom kompartmentu", r"tricompartment", r"three compartments", r"all three compartments", r"tres compartimentos", r"trois compartiments", r"trikompartiment", r"drie compartimenten", r"tri kompartiment", r"трикомпартмент", r"üç kompartman", r"tricompartimental"],
    "lateral_oa": [r"lateral (?:compartment|tibiofemoral|femorotibial|femoral condyle|tibial plateau)", r"compartimento (?:lateral|externo)",
                   r"compartiment (?:externe|fémoro-tibial externe|latéral)", r"lateral\w* kompartiment",
                   r"латеральн\w+ (?:отдел|мыщел)", r"외측 구획", r"外侧间室", r"lateral femoral", r"lateral tibial",
                   r"lateraln\w* (?:kompartment|kondil|tibijaln)", r"έξω (?:διαμέρισμ|μηριαί|κνημιαί)", r"lateral (?:kompartman|femoral|tibial)",
                   r"lateraal", r"tricompartment", r"three compartments", r"all three compartments", r"tres compartimentos", r"trois compartiments", r"trikompartiment", r"drie compartimenten", r"tri kompartiment", r"трикомпартмент", r"üç kompartman", r"tricompartimental"],
    "patellofemoral_oa": [r"patell?o-?femoral", r"fémoro-?patell", r"femoro-?patel", r"femororrotul", r"femoro-?rotule",
                          r"retropatell", r"пателлофеморальн", r"надколенник", r"patella", r"rótula", r"rotule", r"kniescheibe",
                          r"trochlea", r"髌股", r"슬개대퇴", r"patel", r"rotul", r"επιγονατιδ", r"троχλ", r"пател", r"trohle",
                          r"femoropatel", r"patellofemoral", r"tricompartment", r"three compartments", r"all three compartments", r"tres compartimentos", r"trois compartiments", r"trikompartiment", r"drie compartimenten", r"tri kompartiment", r"трикомпартмент", r"üç kompartman", r"tricompartimental"],
    "effusion": [r"effusion", r"derrame", r"épanchement", r"epanchement", r"versamento", r"erguss", r"выпот", r"hydrops",
                 r"gewrichtsvocht", r"płyn w stawie", r"efüzyon", r"eklem sıvı", r"积液", r"삼출", r"관절액", r"hemarthros",
                 r"haemarthros", r"lipohemarthros", r"izljev", r"izliv", r"συλλογ\w* υγρ", r"υγρό", r"излив", r"eklem içi sıvı",
                 r"eklem sıvısı", r"sıvı miktarı", r"\bhydrops", r"vochtcollectie", r"joint fluid"],
    "synovitis": [r"synovit", r"sinovit", r"synovial (?:thicken|prolifer|hypertroph|enhanc)", r"синовит", r"синовиальн\w+ (?:утолщ|пролифер)",
                  r"zapalenie błony maziowej", r"滑膜炎", r"활막염", r"villonodular", r"lipoma arborescens", r"sinovij", r"υμενίτ",
                  r"синовит", r"sinovyal", r"synoviaal", r"synovial\w* (?:verdikking|hypertrofie)"],
    "bakers_cyst": [r"baker", r"popliteal cyst", r"quiste poplíteo", r"kyste poplité", r"cisti poplitea", r"poplitealzyste",
                    r"киста бейкера", r"подколенн\w+ кист", r"torbiel bakera", r"腘窝囊肿", r"贝克囊肿", r"베이커", r"bakerova",
                    r"poplitealn\w* cist", r"κύστη baker", r"ιγνυακ\w* κύστ", r"бейкер", r"popliteal kist", r"bakerse"],
    "contusion": [r"contusion", r"contusão", r"contusión", r"contusione", r"kontusion", r"kontüzyon", r"bone bruise",
                  r"marrow (?:o?edema|signal)", r"bone marrow", r"edema óseo", r"edema osseo", r"œdème osseux", r"oedème osseux",
                  r"knochenmarködem", r"knochenmarksödem", r"контузи", r"отек костного мозга", r"трабекулярн\w+ отек",
                  r"beenmergoedeem", r"obrzęk szpiku", r"kemik iliği ödemi", r"骨挫伤", r"骨髓水肿", r"골타박", r"골수부종",
                  r"koštan\w* edem", r"edem koštane srži", r"kontuzij", r"οστεομυελικ\w* οίδημ", r"οστικ\w* μώλωπ", r"контузи",
                  r"костномозъч\w* едем", r"костен едем", r"kemik ödemi", r"kemik iliği ödem", r"botcontusie", r"beenmergoedeem"],
    "fracture": [r"fractur", r"fratur", r"frattur", r"fraktur", r"перелом", r"fractuur", r"złamani", r"kırı[kğ]", r"骨折", r"골절",
                 r"prijelom", r"prelom", r"κάταγμ", r"фрактур", r"счупван", r"\bfx\b"],
}
_STRUCT_RE = {k: re.compile("|".join(v), re.IGNORECASE) for k, v in STRUCT.items()}
NEEDS_PATHOLOGY_TERM = {"acl", "mcl", "medial_meniscus", "lateral_meniscus"}
NEEDS_OA_TERM = {"medial_oa", "lateral_oa", "patellofemoral_oa"}


@dataclass
class RuleHit:
    label: str
    status: str  # positive | negative | equivocal
    sentence: str


def split_sentences(text: str) -> list[str]:
    return [s.strip() for s in _SENT_SPLIT.split(text or "") if s and s.strip()]


def rule_extract(text: str) -> dict[str, str]:
    """-> {label: positive|negative|equivocal|unknown}; positive wins over negative across sentences."""
    votes: dict[str, set[str]] = {k: set() for k in LABELS}
    for sent in split_sentences(text):
        low = sent.lower()
        for label, rx in _STRUCT_RE.items():
            if not rx.search(low):
                continue
            neg = bool(_NEG_RE.search(low))
            hedge = bool(_HEDGE_RE.search(low))
            if label in NEEDS_PATHOLOGY_TERM:
                path = bool(_TEAR_RE.search(low))
            elif label in NEEDS_OA_TERM:
                path = bool(_OA_RE.search(low))
            else:
                path = True
            if neg and not path:
                votes[label].add("negative")
            elif neg and path:
                votes[label].add("negative")  # "no tear of the ACL" - negation dominates
            elif path and hedge:
                votes[label].add("equivocal")
            elif path:
                votes[label].add("positive")
    out: dict[str, str] = {}
    for label, v in votes.items():
        if "positive" in v:
            out[label] = "positive"
        elif "equivocal" in v:
            out[label] = "equivocal"
        elif "negative" in v:
            out[label] = "negative"
        else:
            out[label] = "unknown"
    return out
