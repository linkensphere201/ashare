from __future__ import annotations

from collections.abc import Callable

import polars as pl

Mapper = Callable[[pl.DataFrame], pl.DataFrame]


def map_tushare_raw_to_curated(frame: pl.DataFrame, dataset: str) -> pl.DataFrame:
    try:
        return TUSHARE_RAW_MAPPERS[dataset](frame)
    except KeyError as error:
        raise ValueError(f"unsupported raw dataset mapping: {dataset}") from error


def _security_master(frame: pl.DataFrame) -> pl.DataFrame:
    return frame.with_columns(
        [
            pl.col("ts_code").alias("symbol"),
            pl.col("symbol").alias("raw_symbol"),
            pl.col("ts_code").str.split(".").list.get(1).alias("exchange"),
            pl.lit("stock").alias("asset_type"),
            _market_segment_expr().alias("market_segment"),
            _market_segment_name_expr().alias("market_segment_name"),
            _parse_yyyymmdd("list_date").alias("list_date"),
            _parse_yyyymmdd("delist_date").alias("delist_date"),
            pl.lit("active").alias("status"),
        ]
    ).select(["symbol", "raw_symbol", "exchange", "asset_type", "name", "market", "market_segment", "market_segment_name", "area", "industry", "list_date", "delist_date", "status"])


def _trading_calendar(frame: pl.DataFrame) -> pl.DataFrame:
    return frame.with_columns(
        [
            pl.lit("cn_a_share").alias("calendar_id"),
            _parse_yyyymmdd("cal_date").alias("trade_date"),
            (pl.col("is_open").cast(pl.Int64, strict=False) == 1).alias("is_trading_day"),
            _parse_yyyymmdd("pretrade_date").alias("previous_trade_date"),
            pl.lit(None).alias("next_trade_date"),
        ]
    ).select(["calendar_id", "trade_date", "is_trading_day", "previous_trade_date", "next_trade_date"])


def _daily_prices(frame: pl.DataFrame) -> pl.DataFrame:
    return frame.with_columns([pl.col("ts_code").alias("symbol"), _parse_yyyymmdd("trade_date").alias("trade_date"), pl.lit("stock").alias("asset_type"), pl.col("vol").alias("volume"), pl.col("pct_chg").alias("pct_change")]).select(["symbol", "trade_date", "asset_type", "open", "high", "low", "close", "pre_close", "volume", "amount", "pct_change"])


def _index_daily(frame: pl.DataFrame) -> pl.DataFrame:
    volume_expr = pl.col("vol").alias("volume") if "vol" in frame.columns else pl.lit(None).alias("volume")
    amount_expr = pl.col("amount").alias("amount") if "amount" in frame.columns else pl.lit(None).alias("amount")
    pct_change_expr = pl.col("pct_chg").alias("pct_change") if "pct_chg" in frame.columns else pl.lit(None).alias("pct_change")
    pre_close_expr = pl.col("pre_close").alias("pre_close") if "pre_close" in frame.columns else pl.lit(None).alias("pre_close")
    return frame.with_columns([pl.col("ts_code").alias("symbol"), _parse_yyyymmdd("trade_date").alias("trade_date"), pl.lit("index").alias("asset_type"), volume_expr, amount_expr, pct_change_expr, pre_close_expr]).select(["symbol", "trade_date", "asset_type", "open", "high", "low", "close", "pre_close", "volume", "amount", "pct_change"])


def _adj_factor(frame: pl.DataFrame) -> pl.DataFrame:
    return frame.with_columns([pl.col("ts_code").alias("symbol"), _parse_yyyymmdd("trade_date").alias("trade_date"), pl.col("adj_factor").cast(pl.Float64, strict=False).alias("adj_factor")]).select(["symbol", "trade_date", "adj_factor"])


def _daily_basic(frame: pl.DataFrame) -> pl.DataFrame:
    return frame.with_columns([pl.col("ts_code").alias("symbol"), _parse_yyyymmdd("trade_date").alias("trade_date"), pl.col("turnover_rate").cast(pl.Float64, strict=False).alias("turnover_rate")]).select(["symbol", "trade_date", "turnover_rate"])


def _stk_limit(frame: pl.DataFrame) -> pl.DataFrame:
    return frame.with_columns([pl.col("ts_code").alias("symbol"), _parse_yyyymmdd("trade_date").alias("trade_date"), pl.col("up_limit").cast(pl.Float64, strict=False).alias("limit_up"), pl.col("down_limit").cast(pl.Float64, strict=False).alias("limit_down")]).select(["symbol", "trade_date", "limit_up", "limit_down"])


def _suspend_d(frame: pl.DataFrame) -> pl.DataFrame:
    return frame.with_columns([pl.col("ts_code").alias("symbol"), _parse_yyyymmdd("trade_date").alias("trade_date"), (pl.col("suspend_type") == "S").alias("is_suspended")]).select(["symbol", "trade_date", "is_suspended"])


def _moneyflow_dc(frame: pl.DataFrame) -> pl.DataFrame:
    return frame.with_columns([pl.col("ts_code").alias("symbol"), _parse_yyyymmdd("trade_date").alias("trade_date"), pl.col("net_amount").alias("main_net_inflow"), pl.col("net_amount_rate").alias("main_net_inflow_rate"), pl.lit("tushare_moneyflow_dc").alias("data_method")]).select(["symbol", "trade_date", "main_net_inflow", "main_net_inflow_rate", "data_method"])


def _cyq_perf(frame: pl.DataFrame) -> pl.DataFrame:
    data_method = pl.when(pl.col("provider_status") == "not_found").then(pl.lit("tushare_cyq_perf:not_found")).otherwise(pl.lit("tushare_cyq_perf")) if "provider_status" in frame.columns else pl.lit("tushare_cyq_perf")
    return frame.with_columns([pl.col("ts_code").alias("symbol"), _parse_yyyymmdd("trade_date").alias("trade_date"), pl.col("winner_rate").alias("close_profit_ratio"), data_method.alias("data_method")]).select(["symbol", "trade_date", "close_profit_ratio", "data_method"])


def _index_classify(frame: pl.DataFrame) -> pl.DataFrame:
    parent_expr = pl.col("parent_code") if "parent_code" in frame.columns else pl.lit(None)
    src_expr = pl.col("src") if "src" in frame.columns else pl.lit("SW2021")
    return frame.with_columns([pl.col("index_code").alias("index_code"), pl.col("industry_name").alias("industry_name"), pl.col("level").alias("level"), src_expr.alias("source_system"), parent_expr.alias("parent_code")]).select(["index_code", "industry_name", "level", "source_system", "parent_code"])


def _sw_daily(frame: pl.DataFrame) -> pl.DataFrame:
    name_expr = pl.col("name").alias("industry_name") if "name" in frame.columns else pl.lit(None).alias("industry_name")
    open_expr = pl.col("open").alias("open") if "open" in frame.columns else pl.lit(None).alias("open")
    high_expr = pl.col("high").alias("high") if "high" in frame.columns else pl.lit(None).alias("high")
    low_expr = pl.col("low").alias("low") if "low" in frame.columns else pl.lit(None).alias("low")
    volume_expr = pl.col("vol").alias("volume") if "vol" in frame.columns else pl.lit(None).alias("volume")
    amount_expr = pl.col("amount").alias("amount") if "amount" in frame.columns else pl.lit(None).alias("amount")
    pct_change_expr = pl.col("pct_change").alias("pct_change") if "pct_change" in frame.columns else pl.col("pct_chg").alias("pct_change")
    pre_close_expr = pl.col("pre_close").alias("pre_close") if "pre_close" in frame.columns else pl.lit(None).alias("pre_close")
    return frame.with_columns([pl.col("ts_code").alias("index_code"), _parse_yyyymmdd("trade_date").alias("trade_date"), name_expr, open_expr, high_expr, low_expr, pct_change_expr, pre_close_expr, volume_expr, amount_expr]).select(["index_code", "trade_date", "industry_name", "open", "high", "low", "close", "pre_close", "pct_change", "volume", "amount"])


TUSHARE_RAW_MAPPERS: dict[str, Mapper] = {
    "security_master": _security_master,
    "trading_calendar": _trading_calendar,
    "daily_prices": _daily_prices,
    "index_daily": _index_daily,
    "adj_factor": _adj_factor,
    "daily_basic": _daily_basic,
    "stk_limit": _stk_limit,
    "suspend_d": _suspend_d,
    "moneyflow_dc": _moneyflow_dc,
    "cyq_perf": _cyq_perf,
    "index_classify": _index_classify,
    "sw_daily": _sw_daily,
}


def _parse_yyyymmdd(column: str) -> pl.Expr:
    if column in ("delist_date", "pretrade_date"):
        return pl.when(pl.col(column).is_null() | (pl.col(column).cast(pl.Utf8) == "")).then(None).otherwise(pl.col(column).cast(pl.Utf8).str.strptime(pl.Date, "%Y%m%d", strict=False))
    return pl.col(column).cast(pl.Utf8).str.strptime(pl.Date, "%Y%m%d", strict=False)


def _market_segment_expr() -> pl.Expr:
    exchange = pl.col("ts_code").str.split(".").list.get(1)
    return (
        pl.when((exchange == "SH") & (pl.col("market") == "主板")).then(pl.lit("sh_main"))
        .when((exchange == "SH") & (pl.col("market") == "科创板")).then(pl.lit("star"))
        .when((exchange == "SZ") & (pl.col("market") == "主板")).then(pl.lit("sz_main"))
        .when((exchange == "SZ") & (pl.col("market") == "创业板")).then(pl.lit("chinext"))
        .when((exchange == "BJ") | (pl.col("market") == "北交所")).then(pl.lit("bj"))
        .otherwise(pl.lit("unknown"))
    )


def _market_segment_name_expr() -> pl.Expr:
    exchange = pl.col("ts_code").str.split(".").list.get(1)
    return (
        pl.when((exchange == "SH") & (pl.col("market") == "主板")).then(pl.lit("上证主板"))
        .when((exchange == "SH") & (pl.col("market") == "科创板")).then(pl.lit("科创板"))
        .when((exchange == "SZ") & (pl.col("market") == "主板")).then(pl.lit("深市主板"))
        .when((exchange == "SZ") & (pl.col("market") == "创业板")).then(pl.lit("创业板"))
        .when((exchange == "BJ") | (pl.col("market") == "北交所")).then(pl.lit("北交所"))
        .otherwise(pl.lit("未知"))
    )
