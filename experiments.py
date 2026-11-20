import json
import sys
from pathlib import Path

from train import run

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


def value_keys(df):
    return df.copy()


def digit_keys(df):
    out = value_keys(df)
    for c in DECIMALS:
        cents = (df[c] * 100).round().astype("int64")
        out[f"{c}_int"] = cents // 100
        out[f"{c}_int10"] = cents // 1000
        out[f"{c}_d1"] = cents // 10 % 10
        out[f"{c}_cents"] = cents % 100
        out[f"{c}_lead"] = cents.astype(str).str[0]
    return out


VARIANTS = {
    "baseline": {},
    "flag_count": {"prep": flag_count},
    "as_category": {"prep": as_category},
    "drop_noise": {"prep": drop_noise},
    "all_three": {"prep": lambda df: drop_noise(as_category(flag_count(df)))},
    "te_values": {"keys": value_keys},
    "te_digits": {"keys": digit_keys},
    "glm_offset": {"keys": digit_keys, "glm": True},
    "te_freq": {"keys": digit_keys, "prep": frequency},
    "te_plain": {"keys": digit_keys, "prep": plain_digits},
    "te_freq_plain": {"keys": digit_keys, "prep": lambda df: plain_digits(frequency(df))},
}

if __name__ == "__main__":
    comp, params_file, *names = sys.argv[1:]
    params = json.loads(Path(params_file).read_text())
    for name in names or VARIANTS:
        score, _ = run(Path("data") / comp, params=params, **VARIANTS[name])
        print(f"{name:<12} CV {score:.5f}", flush=True)
