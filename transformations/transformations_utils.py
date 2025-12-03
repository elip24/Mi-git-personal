import html
import re
from urllib.parse import unquote
import polars as pl
import unicodedata

def verify_mergers(df):
    check_type = pl.col("type").cast(pl.Utf8).str.strip_chars().str.to_lowercase()
    has_non_deal = df.select(check_type.ne("deal brief").fill_null(True).any()).item()
    if has_non_deal:
        print("Error: check the site's API")


def get_from_string_to_date(df:pl.DataFrame,column_name:str):
    df=df.with_columns(pl.col((f"{column_name}")).str.strptime(pl.Datetime, "%Y-%m-%d").dt.date()
                       .alias(column_name))
    return df

def clean_string(s:str):
    s = unquote(s)
    s=html.unescape(s)
    s=unicodedata.normalize("NFC",s)
    s=unicodedata.normalize("NFKC", html.unescape(s))
    return s

def clean_string_columns(df: pl.DataFrame) -> pl.DataFrame:
    text_cols = [c for c, t in df.schema.items() if t == pl.Utf8]
    return df.with_columns(
        [
            pl.col(c)
            .map_elements(clean_string, return_dtype=pl.Utf8)
            .alias(c)
            for c in text_cols
        ]
    )


def transform_array_into_list(df:pl.DataFrame,column_name):
    df = df.with_columns(
    pl.when(pl.col((f"{column_name}")).is_null())
      .then(pl.lit([]).cast(pl.List(pl.Utf8)))
      .otherwise(pl.col((f"{column_name}")).map_elements(
        lambda xs: [clean_string(x) for x in xs], return_dtype=pl.List(pl.Utf8) #Because sometimes you could get yeñen or weird name due to html
    ))
      .alias((f"{column_name}"))
    )
    return df

def drop_if_list_only_null(df:pl.DataFrame):
    for c in df.columns:
        if df.schema[c]==pl.List and df.select((pl.col(c).list.len().fill_null(0).sum() == 0)).item():
            df = df.drop([c])
    return df

def make_hash_col(df:pl.DataFrame,key_cols:list):
    expresion=list()
    for c in df.columns:
        if c not in key_cols:
            continue
        if df[c].dtype==pl.List:
            e=pl.col(c)
            e=e.list.unique(maintain_order=True)
            e=e.list.join(",").fill_null("")
        else:
            e=pl.col(c).cast(pl.Utf8).fill_null("")
        expresion.append(e)
    df=df.with_columns(pl.concat_str(expresion,separator="||").hash(seed=0).cast(pl.Utf8).alias("hash_cols"))
    return df

def get_from_string_to_datetime(df:pl.DataFrame,column_name:str,alias_name:str):
    df=df.with_columns(pl.col((f"{column_name}")).str.strptime(pl.Datetime, "%Y-%m-%d").dt.date()
                       .alias(alias_name))
    return df


def clean_text_for_ml(df:pl.DataFrame,column_name:str):
    remove_all_weird_characters = (
        pl.col(column_name)
        .str.replace_all(r"-\s*\r?\n\s*", "") #Only if we have one half of a word in one pharagraph and the rest of the word in the other
        .str.replace_all("\r\n", "\n")
        .str.replace_all("\u00A0", " ")  # NBSP -> space
        .str.replace_all(r"\n{2,}", ". ")  # pharagraps (so that everything is one long line for my ml)
        .str.replace_all(r"\n", " ")  # \n -> space
        .str.replace_all(r"[ \t]+", " ")  # colpases tabs
        .str.strip_chars()
    )
    needs_udf = (
            remove_all_weird_characters.str.contains(r"&[A-Za-z0-9#]+;") | #HTML propertys
            remove_all_weird_characters.str.contains(r"[^\x00-\x7F]") #no-ASCII
    )

    df=df.with_columns(pl.when(needs_udf).then(
            remove_all_weird_characters.map_elements(
                lambda s: unicodedata.normalize("NFKC", html.unescape(s)) if s else s,
                return_dtype=pl.Utf8,
            )
        ).otherwise(remove_all_weird_characters)
                       .alias(f"clean_{column_name}")
    )
    return df

def normalize_person_name(s: str | None) -> str | None:
    s = html.unescape(s)
    s = unicodedata.normalize("NFD", s)
    s = "".join(ch for ch in s if unicodedata.category(ch) != "Mn")
    s = s.replace("’","'").replace("`","'")
    s = re.sub(r"[\u2010-\u2015\-]+", " ", s)
    s = s.replace("'", "")
    s = s.lower()
    s = re.sub(r"[^a-z\s-]", " ", s)
    s = re.sub(r"\s+", " ", s).strip()
    return s
