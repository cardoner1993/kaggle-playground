import sys
from pathlib import Path

from train import run

FLAGS = ["is_foreign_transaction", "is_new_merchant", "used_vpn", "ip_country_mismatch",
         "billing_shipping_mismatch", "is_ai_generated_scam_attempt"]
NOISE = ["card_type", "channel", "device_type"]
AS_CAT = ["customer_age", "hours_since_last_txn"]


def flag_count(df):
    return df.assign(risk_flags=df[FLAGS].sum(axis=1) + (df.auth_method == "No Authentication"))


def as_category(df):
    return df.assign(**{f"{c}_cat": df[c].astype(str) for c in AS_CAT})


def drop_noise(df):
    return df.drop(columns=NOISE)


VARIANTS = {
    "baseline": lambda df: df,
    "flag_count": flag_count,
    "as_category": as_category,
    "drop_noise": drop_noise,
    "all_three": lambda df: drop_noise(as_category(flag_count(df))),
}

if __name__ == "__main__":
    comp = sys.argv[1] if len(sys.argv) > 1 else "playground-series-s6e11"
    for name in sys.argv[2:] or VARIANTS:
        score, _ = run(Path("data") / comp, prep=VARIANTS[name])
        print(f"{name:<12} CV {score:.5f}", flush=True)
