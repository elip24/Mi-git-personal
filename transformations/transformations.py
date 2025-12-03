import polars as pl
import json
from datetime import datetime, timezone

from src.transformations.tranformations_ml import extract_people_from_text
from src.transformations.transformations_utils import get_from_string_to_date, transform_array_into_list, make_hash_col, \
    clean_text_for_ml, normalize_person_name, clean_string_columns, verify_mergers, drop_if_list_only_null


def transformations_news_raw(all_news:dict):
    df = pl.DataFrame(all_news)
    df=df.select('headline','datePublished','lawyer_names','lawyer_link','text','url')
    df=drop_if_list_only_null(df)
    df=df.with_columns(pl.lit(datetime.now(timezone.utc)).cast(pl.Datetime).alias('syncstartdatetime'))
    return df

def transformations_news(all_news:dict)->pl.DataFrame:
    df = pl.DataFrame(all_news)
    df=df.unique('url')
    df=clean_string_columns(df)
    df=transform_array_into_list(df,"capabilities")
    df = transform_array_into_list(df, "lawyer_link")
    df = transform_array_into_list(df, "lawyer_names")
    df = df.with_columns(pl.lit('cleary').alias('site_page'))
    df=df.with_columns(pl.col('url').str.split('/').list.last().alias('id'))
    df = make_hash_col(df,key_cols=['id','site_page','url'])
    df = df.with_columns(pl.lit(datetime.now(timezone.utc)).cast(pl.Datetime).alias("syncstartdatetime"))

    return df

base_cols = ["id", "site_page", "datePublished", "headline", "url", "syncstartdatetime"]

def transformations_news_per_person(df):
    df = clean_text_for_ml(df, column_name='text')
    df = extract_people_from_text(df, text_col='clean_text', out_names=("persons_ml", "score_ml"), min_conf=0)
    df = df.with_columns(pl.col("persons_ml").list
                         .eval(pl.element().map_elements(normalize_person_name, return_dtype=pl.Utf8))
                         .alias("persons_ml_norm"))
    link_rows = (
        df.select(base_cols + ["lawyer_link"])
        .explode("lawyer_link")
        .filter(pl.col("lawyer_link").is_not_null())
        .with_columns([
            pl.lit("link_html").alias("match_source"),
            pl.lit(1.0).alias("score_ml"),
            pl.lit(None, dtype=pl.Utf8).alias("persons_ml"),
            pl.lit(None, dtype=pl.Utf8).alias("persons_ml_norm"),
        ])
    )
    ml_rows = df.select(base_cols + ["persons_ml", "persons_ml_norm", "score_ml"]) \
        .explode(["persons_ml", "persons_ml_norm", "score_ml"]) \
        .filter(pl.col("persons_ml").is_not_null()) \
        .with_columns([
            pl.lit(None, dtype=pl.Utf8).alias("lawyer_link"),
            pl.lit("body_ml").alias("match_source"),
        ]) \
        .select(base_cols + ["lawyer_link", "match_source", "score_ml", "persons_ml", "persons_ml_norm"])
    df = pl.concat([link_rows, ml_rows], how="vertical").rename({"id": "article_id"})
    return df

def transformations(all_news:dict):
    df_clean = transformations_news(all_news)
    df_raw=transformations_news_raw(all_news)
    df_line=transformations_news_per_person(df_clean)
    return df_clean,df_raw,df_line