from pathlib import Path
import pandas as pd
from dirty_da.schema_semantics import load_table_semantics, validate_table_semantics

ROOT = Path(__file__).resolve().parents[1]
TABLES = ROOT / "data" / "tables"
paths = sorted(TABLES.glob("*.csv"))
if not paths:
    raise RuntimeError(f"No CSV tables found in {TABLES}")
registry = load_table_semantics(ROOT / "configs" / "table_semantics.json")
if set(registry["tables"]) != {path.name for path in paths}:
    raise RuntimeError("Table semantics registry does not match the CSV inventory.")
for path in paths:
    df = pd.read_csv(path, low_memory=False)
    roles = validate_table_semantics(df, path.name, registry)
    identifiers = sum(role == "identifier" for role in roles.values())
    targets = len(registry["tables"][path.name].get("targets", []))
    replacement_chars = sum(
        int(df[column].astype("string").str.contains("\ufffd", regex=False, na=False).sum())
        for column in df.select_dtypes(include=["object", "string"]).columns
    )
    print(
        f"{path.name}\trows={len(df)}\tcols={len(df.columns)}"
        f"\tmissing={int(df.isna().sum().sum())}\tdups={int(df.duplicated().sum())}"
        f"\tidentifiers={identifiers}\ttargets={targets}"
        f"\treplacement_chars={replacement_chars}\tsemantics=PASS"
    )
