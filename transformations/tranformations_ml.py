import re
import spacy
from spacy.cli import download
import polars as pl

def load_spacy():
        # load only  NER; exclude everything else
    nlp = spacy.load(
            "en_core_web_sm",
            exclude=["tok2vec","tagger","morphologizer","parser","attribute_ruler","lemmatizer","senter"]
        )

        # segmentation of sentences
    nlp.add_pipe(
        "sentencizer",
        first=True,
        config={"punct_chars": [".","!","?","\n",";"]}  # we could add  "•", ":"
    )
    # since we are reading articles, we are going to put the maximum length just in case
    nlp.max_length = max(nlp.max_length, 2_000_000)
    return nlp


nlp = load_spacy()
ROLE_HINT = re.compile(r'\b(partner|partners|associate|associates|counsel|led by|team|director|directors)\b', re.I)
ALIAS_LAST = re.compile(r'\b([A-Z]{2,4})\s+([A-Z][a-zÀ-ÖØ-öø-ÿ][A-Za-zÀ-ÖØ-öø-ÿ\'\-]+)\b')
PAREN_NICK = re.compile(
    r'\b([A-Z][a-zÀ-ÖØ-öø-ÿ\'\-]+)\s*\(\s*([A-Za-z.]{1,10})\s*\)\s+([A-Z][a-zÀ-ÖØ-öø-ÿ\'\-]+)\b'
)
BAD_UPPER = {'PC', 'GMBH','BV', 'NV', 'SPL', 'PLC', 'S.P.A', 'LLP', 'LLC', 'INC', 'SA', 'SAS', 'AIIB', 'US', 'UAE', 'UK', 'USA', 'ESG', 'IPO', 'M&A','GLOBAL','WORLD',
             'SOFTWARE'}

def has_role_hint_around(ent, text, radius=120) -> bool:
    i, j = ent.start_char, ent.end_char
    seg = text[max(0, i - radius): min(len(text), j + radius)]
    return ROLE_HINT.search(seg) is not None

def _canon(n: str) -> str:
    n = re.sub(r'\s+', ' ', n).strip()
    return n.replace("’", "'")

def extract_people_in_roles(text: str) -> list[str]:
    doc = nlp(text or "")
    names = []
    alias_map = {}  # alias(lower) -> canonical
    for ent in doc.ents:
        if ent.label_ == "PERSON" and " " in ent.text.strip() :
            names.append(ent.text.strip())

    for sent in doc.sents:
        s = sent.text
        if not ROLE_HINT.search(s):
            continue

        #Not likely to happen (this is for when we have Kenneth (KC) Sands)
        for first, nick, last in PAREN_NICK.findall(s):
            nick_clean = nick.replace('.', '')
            canonical = f"{first} ({nick_clean}) {last}"
            alias = f"{nick_clean.upper()} {last}"
            first_last = f"{first} {last}"
            names.extend([canonical, alias, first_last])
            alias_map[alias.lower()] = canonical
            alias_map[first_last.lower()] = canonical

        # This is when we have an alias as a short name + surname (example:Kc sands where is a person not a firm name)
        for up_alias, last in ALIAS_LAST.findall(s):
            alias = up_alias.replace('.', '')
            if alias in BAD_UPPER:
                continue
            names.append(f"{alias} {last}")

    seen, out = set(), []
    for n in names:
        k = n.lower()
        resolved = alias_map.get(k, n)
        rk = _canon(resolved).lower()
        if rk not in seen:
            seen.add(rk)
            out.append(resolved)
    return out

def _shape_score(name: str) -> float:
    toks = name.split()
    if len(toks) < 2:
        return 0.0
    base = 0.7 if 2 <= len(toks) <= 4 else 0.45
    if any(ch.isdigit() for t in toks for ch in t):
        base -= 0.2
    if any("&" in t for t in toks):
        base -= 0.2
    return max(0.0, min(1.0, base))

W_ROLE = 0.60
W_SHAPE = 0.35
BONUS_ALIAS = 0.05


def extract_people_scored(text: str) -> list[dict]:
    names = extract_people_in_roles(text or "")
    out = []
    for n in names:
        name = n.strip()
        if " " not in name:
            continue
        role = 1.0  # we tell if it has role its a confidence of 100
        shape = _shape_score(name)
        alias_bonus = BONUS_ALIAS if ("(" in name and ")" in name) else 0.0
        conf = max(0.0, min(1.0, round(W_ROLE*role + W_SHAPE*shape + alias_bonus, 3)))
        out.append({"person": name, "confidence": conf})
    return out

RET_SCORED = pl.List(pl.Struct([pl.Field("person", pl.Utf8),
                                pl.Field("confidence", pl.Float64)]))

def extract_people_from_text(df: pl.DataFrame, text_col, out_names: tuple[str,str],min_conf:float,) -> pl.DataFrame:
    names_col, scores_col = out_names
    scored_expr = pl.col(text_col).map_elements(extract_people_scored, return_dtype=RET_SCORED)

    if min_conf > 0:
        scored_expr = scored_expr.list.eval(
            pl.when(pl.element().struct.field("confidence") >= pl.lit(min_conf))
            .then(pl.element())
        ).list.drop_nulls()
    df = df.with_columns(scored_expr.alias("_ml_scored"))

    df = df.with_columns([
        pl.col("_ml_scored").list.eval(pl.element().struct.field("person")).alias(names_col),
        pl.col("_ml_scored").list.eval(pl.element().struct.field("confidence")).alias(scores_col),
    ])

    df = df.drop("_ml_scored")
    return df