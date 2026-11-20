import sys
import tomllib
from pathlib import Path

from train import digit_keys, run, value_keys

FLAGS = ["is_foreign_transaction", "is_new_merchant", "used_vpn", "ip_country_mismatch",
         "billing_shipping_mismatch", "is_ai_generated_scam_attempt"]
NOISE = ["card_type", "channel", "device_type"]
AS_CAT = ["customer_age", "hours_since_last_txn"]
DECIMALS = ["amount_usd", "hours_since_last_txn", "distance_from_home_km",
            "account_balance_usd", "velocity_score", "merchant_risk_score"]


def flag_count(df):
    return df.assign(risk_flags=df[FLAGS].sum(axis=1) + (df.auth_method == "No Authentication"))


def as_category(df):
    return df.assign(**{f"{c}_cat": df[c].astype(str) for c in AS_CAT})


def drop_noise(df):
    return df.drop(columns=NOISE)


def frequency(df):
    num = df.select_dtypes("number").columns
    return df.assign(**{f"{c}_freq": df[c].map(df[c].value_counts(normalize=True)) for c in num})


def plain_digits(df):
    cents = {c: (df[c] * 100).round().astype("int64") for c in DECIMALS}
    return df.assign(**{f"{c}_d1n": v // 10 % 10 for c, v in cents.items()},
                     **{f"{c}_centsn": v % 100 for c, v in cents.items()})


def digits(df):
    return digit_keys(df, DECIMALS)


VARIANTS = {
    "baseline": {},
    "flag_count": {"prep": flag_count},
    "as_category": {"prep": as_category},
    "drop_noise": {"prep": drop_noise},
    "all_three": {"prep": lambda df: drop_noise(as_category(flag_count(df)))},
    "te_values": {"keys": value_keys},
    "te_digits": {"keys": digits},
    "glm_offset": {"keys": digits, "glm": True},
    "te_freq": {"keys": digits, "prep": frequency},
    "te_plain": {"keys": digits, "prep": plain_digits},
    "te_freq_plain": {"keys": digits, "prep": lambda df: plain_digits(frequency(df))},
}

if __name__ == "__main__":
    config, *names = sys.argv[1:]
    cfg = tomllib.loads(Path(config).read_text())
    for name in names or VARIANTS:
        score, _ = run(Path("data") / cfg["comp"], params=cfg["params"],
                       tags={"model": cfg["model"], "variant": name}, **VARIANTS[name])
        print(f"{name:<12} CV {score:.5f}", flush=True)
