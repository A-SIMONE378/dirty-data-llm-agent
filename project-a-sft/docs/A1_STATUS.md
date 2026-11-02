# A1 Table Selection Audit

## Verdict

**A1 table selection passes.** The catalog contains 40 selected core tables and
5 selected domain-generalization tables. The core tables are partitioned by
table into 28 train, 4 development, and 8 test tables, exactly matching the
declared 70/10/20 policy. Nine tables are retained in quarantine and are not
part of any formal split: seven have unresolved source/license evidence, and
two failed the semantic admission rule during the A2 review.

This stage is CPU-only. No model, CUDA, PyTorch, Transformers, PEFT, or training
runtime is required.

## Inventory and provenance

- Total CSV files cataloged: 54
- Selected and research-approved: 45
- Quarantined: 9
- Selected domains recorded: 36
- Core split: 28 train, 4 development, 8 test
- Domain-generalization holdout: 5

The original pack contributed four selected UCI-derived tables. Their official
UCI source and CC BY 4.0 license are recorded, while their original 5000-row
selection procedure remains marked as partial provenance.

The A1 import plan added 41 tables directly from official UCI endpoints. For
each table, the importer verifies the official dataset page contains CC BY 4.0
license text, preserves UCI variable names and values, records the DOI and
source URL, and writes a SHA-256 hash to the import manifest.

Five sources exceeded 5000 rows and were reduced deterministically:

- MAGIC Gamma Telescope: 19,020 to 5,000 rows
- Bank Marketing: 45,211 to 5,000 rows
- Chess King-Rook versus King: 28,056 to 5,000 rows
- AI4I Predictive Maintenance: 10,000 to 5,000 rows
- Steel Industry Energy Consumption: 35,040 to 5,000 rows

The selected source row positions are those with the smallest SHA-256 rank of
`uci_id:source_row_position`, then restored to source order. This avoids a
version-dependent random sample.

Thirteen imported tables contained duplicate source rows. They retain every
source row and receive a deterministic `row_id` derived from UCI id and original
source position. No duplicate was silently dropped.

## Quarantine boundary

The following original pilot files remain available for auditing but cannot be
used for formal task generation, training, evaluation, or release until their
source and permission evidence is completed:

- `china_city_statistics.csv`
- `customer_master.csv`
- `sales_transactions.csv`
- `student_grade_ses.csv`
- `student_sleep_screen_time.csv`
- `tianchi_baby_trade_history.csv`
- `zomato_restaurants.csv`

Catalog validation forbids an unknown-license or quarantine record from being
marked research-approved.

The A2 semantic review also quarantined two official UCI tables without
deleting them:

- `uci_247_istanbul_stock_exchange.csv`: `SP` is a continuous return, not a
  category, so the table has no true categorical field.
- `uci_536_pedestrians_traffic.csv`: angles are continuous and the `other_*`
  fields are list-like/opaque, so the table has no scalar categorical field.

They were replaced in the train split by `uci_159_magic_gamma.csv` and
`uci_222_bank_marketing.csv`, preserving all split counts.

## Field semantics

Every column in all 54 CSV files has exactly one primary role: identifier,
category, date, measure, derived, coordinate, or opaque. Targets are tagged
separately. UCI roles and variable types are used as the source of field
semantics. Numeric storage alone never promotes IDs, binary or categorical
codes, coordinates, or derived values into generic measures.

Source metadata was checked against actual values. The reviewed corrections
also classify computer model names as identifiers, currency/percentage strings
as measures, portfolio flags as categories, and free-form drug-review text as
opaque. `scripts/apply_a1_semantics_corrections.py` makes these corrections
explicit and reproducible.

## Corruption reproducibility

The committed low, medium, and high profiles were applied to every table. For
each of 54 tables and 3 profiles, the audit generated the dirty copy twice and
compared metadata and fingerprints.

- Dirty variants checked: 162
- Reproducible variants: 162
- Canonical integrity checks passed: 162
- Failed variants: 0

Text-format and text-category corruptions remain explicitly inapplicable on
numeric-coded category tables. Such cases are recorded, not counted as fake
successful pollution.

## Reproducible files

- Selection catalog: `configs/a1_table_catalog.json`
- Import plan: `configs/a1_uci_import_plan.json`
- Corruption profiles: `configs/a1_corruption_profiles.json`
- Column semantics: `configs/table_semantics.json`
- Importer: `scripts/import_uci_a1_tables.py`
- Candidate discovery: `scripts/discover_uci_candidates.py`
- Selection audit: `scripts/run_a1_selection_audit.py`
- Import manifest: `artifacts/a1_uci_import_manifest.csv`
- Catalog audit: `artifacts/a1_table_catalog_audit.csv`
- Corruption audit: `artifacts/a1_corruption_reproducibility.csv`
- Summary: `artifacts/a1_selection_summary.json`

## Acceptance commands

```bash
python -m pytest -v
python scripts/validate_data_pack.py
python scripts/run_a0_data_audit.py
python scripts/run_a1_selection_audit.py
```

The optional one-time UCI import dependency is installed with:

```bash
python -m pip install -e ".[data]"
python scripts/import_uci_a1_tables.py
```

The importer is idempotent: an existing file must match the regenerated UTF-8
bytes exactly or the command refuses to overwrite it.

## Known limitations and next gate

- The nine quarantined tables are not evidence for A1 scale and are excluded
  from all formal splits.
- Four original UCI-derived files have verified source/license but only partial
  transformation provenance. Their committed canonical CSVs remain immutable.
- Some official source tables contain genuine missing values. These remain part
  of canonical data and must not be confused with subsequently injected missing
  corruption.
- The domain labels and table assignment are now frozen for A1. Moving a table
  between splits requires an explicit catalog change and full audit rerun.

A2 task synthesis and its mechanical audit are now implemented. The next gate
is the independent human review of the 100-question sample documented in
`docs/A2_STATUS.md`. GPU is still unnecessary until later model inference and
LoRA SFT.
