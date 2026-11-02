# A0 Correctness Audit

Audit date: 2026-10-21  
Project: `/root/project-a`  
Environment: Conda `project-a`, Python 3.11.17

## Verdict

**A0 passes the current correctness gate.** The server-side test suite completed
with **52 passed, 0 failed, 0 skipped**. All 11 supplied CSV files have exact
column coverage in the curated semantics registry. The data audit completed 66
corruption applicability checks with 63 execution smoke passes, 3 explicit
inapplicable results, and 0 failures; all 19 task execution smoke checks passed.

This verdict covers the A0 data foundation only. It does not certify model
training, A1 task generation, SFT, GRPO, or generalization to unreviewed tables.

## Scope and method

The audit reviewed the implementation and tests for:

- `CanonicalTable` defensive copying, supported-value validation, fingerprinting,
  and tamper detection;
- all six corruption types, applicability checks, event metadata,
  reproducibility, and ordered compound corruptions;
- explicit-rule reference cleaning without invented recovery;
- `TaskSpec` validation and rejection of ignored or ambiguous fields;
- executor behavior for filters, duplicate handling, sentinels, missing values,
  scalar aggregation, and grouped aggregation;
- verifier behavior for types, floating tolerance, missing results, lists, and
  grouped dictionaries;
- every column in the 11 CSV files, with primary semantic roles and separate
  target tags;
- independence of handwritten expected outputs from the implementation under
  test.

The pre-audit suite had 31 passing tests. Correctness fixes and regression tests
increased the suite to 52 tests.

## Correctness problems found and fixed

1. **Audit answer leakage / self-verification.** The prior audit compared an
   executor result with itself, which could only check output shape. The script
   now reports execution smoke checks only and identifies handwritten pytest
   fixtures as the Ground Truth correctness gate.
2. **Wrong default sentinel for numeric columns.** The column was converted to
   `object` before the default sentinel was selected, causing a string sentinel
   to be used for numeric data. Sentinel selection now uses the original dtype;
   numeric columns receive `-999999` unless an explicit value is supplied.
3. **Invalid duplicate row positions were silently clamped.** They are now
   rejected. Duplicate injection preserves all existing index labels and appends
   an exact row copy instead of resetting the complete index.
4. **Metadata could misrepresent structured values.** Before/after values are
   now JSON-safe structured data rather than Python `repr` strings. Every event
   records an ordered event id, type, severity, seed, row position/index,
   column, operation, and before/after value.
5. **No ordered compound-corruption API.** `apply_corruptions` now applies a
   declared event list in order. Tests cover different columns in the same row,
   an incompatible second corruption on the same cell, and order-dependent
   results.
6. **Datetime format corruption could be silently coerced back by pandas.** A
   corrupted cell is stored in an object column, so the visible format change is
   real. Recovery is possible only with an explicit `datetime_parse` rule and
   declared format.
7. **Task specifications accepted ignored or ambiguous fields.** Validation now
   rejects irrelevant target/group/aggregation fields, empty or malformed
   sentinel maps, invalid `in` filters, duplicate column lists, and malformed
   count tasks.
8. **`missing_policy="error"` inspected unrelated columns.** It now checks only
   columns referenced by the task semantics. A missing value in an unused
   column no longer rejects an otherwise valid task.
9. **`ne` filters admitted missing operands.** Missing operands are now excluded
   consistently rather than satisfying `NaN != value`.
10. **All-missing sums produced a false zero.** Scalar and grouped sums use an
    explicit minimum count and return the missing-result representation rather
    than claiming zero evidence. Missing group keys are excluded under the
    `exclude` policy.
11. **Verifier type and NaN handling was too permissive.** Integer and floating
    results are distinguished, NaN is not accepted as a successful missing
    result (the contract uses `None`), and tolerance values are validated.
12. **Generic audit tasks used dtype/name heuristics.** A curated table-semantics
    registry now prevents identifiers, category codes, coordinates, targets,
    and derived columns from being selected merely because they have numeric
    storage.
13. **Canonical fingerprints accepted ambiguous values.** Duplicate columns and
    unsupported nested mutable values now fail loudly. Fingerprints cover values,
    ordered column labels, dtypes (including categorical metadata), index values,
    index type/name/dtype, and the column-index name.
14. **Cleaner rule shapes were weakly validated.** Rules now reject ambiguous
    structures. The cleaner still does not guess sentinels or reconstruct lost
    source values.

## Table semantics audit

Every CSV column has exactly one primary role from `identifier`, `category`,
`date`, `measure`, `derived`, `coordinate`, or `opaque`. Targets are an
independent tag, not a substitute for a primary role.

| Table | Important identifiers/categories excluded from aggregation | Target or derived fields |
|---|---|---|
| `china_city_statistics.csv` | province, city, region, year | per-capita GDP is derived |
| `customer_master.csv` | customer id and postal code | none |
| `sales_transactions.csv` | transaction/order/customer ids and all coded categories | sales amount and profit are source-derived targets |
| `student_grade_ses.csv` | demographic/school categories and survey date | final grade target |
| `student_sleep_screen_time.csv` | demographic/device categories | sleep-quality and CGPA targets |
| `tianchi_baby_trade_history.csv` | user id, auction id, `cat_id`, `cat1`; `property` is opaque | purchase quantity target |
| `uci_adult.csv` | nominal fields including income | income target; education number is a derived encoding |
| `uci_bike_sharing.csv` | numeric-coded season/year/month/hour/holiday/weekday/weather fields | total rentals is a derived target |
| `uci_seoul_bike.csv` | hour/season/holiday/functioning-day categories | rented-bike count target |
| `uci_wine_quality.csv` | wine color | quality target |
| `zomato_restaurants.csv` | restaurant id/name, country code, price range and other categories; coordinates are not generic measures | aggregate rating target |

The exact machine-readable definitions are in
`configs/table_semantics.json`. Tests require exact registry-to-CSV column
coverage and prevent known numeric identifiers, category codes, targets,
coordinates, and derived fields from entering generic measure selection.

## Ground Truth independence

- Cleaner fixtures contain literal handwritten dirty inputs and expected clean
  outputs. Expected values are not produced by `reference_clean`.
- Executor tests use literal hand-calculated answers, including scalar and
  grouped examples.
- A combined handwritten fixture checks exact-duplicate removal, explicit
  `-999` sentinel mapping, missing exclusion, and a final sum of `10` without
  invoking another implementation to create the expected answer.
- Boundary tests independently assert all-missing sums, unrelated missing
  columns, missing group keys, `ne` with missing operands, and strict verifier
  behavior.
- The bulk audit does not promote smoke-test execution into Ground Truth.

## Corruption and cleaning conclusions

The six supported corruption types are missing values, sentinel values,
duplicate rows, type corruption, format inconsistency, and category
inconsistency. Injection always starts from a defensive copy of canonical data.
Inapplicable requests are rejected or recorded as structured inapplicable
results; they are not counted as successful corruption.

Cleaning is deliberately rule-driven. Duplicate removal, sentinel mapping,
string trimming, category aliases, numeric coercion, and datetime parsing occur
only when explicitly requested. Missing source information is never guessed.

The three semantically honest inapplicable bulk checks were:

- format inconsistency on `tianchi_baby_trade_history.csv`: no suitable free-text
  or supported datetime candidate;
- category inconsistency on `tianchi_baby_trade_history.csv`: categories are
  numeric-coded or opaque rather than free-text categories;
- category inconsistency on `uci_bike_sharing.csv`: categories are numeric-coded.

## Files changed

- Core: `src/dirty_da/__init__.py`, `canonical.py`, `corruption.py`,
  `reference_clean.py`, `task_spec.py`, `executor.py`, `verifier.py`
- New semantics module: `src/dirty_da/schema_semantics.py`
- New registry: `configs/table_semantics.json`
- Scripts: `scripts/validate_data_pack.py`,
  `scripts/run_a0_data_audit.py`
- Tests: `tests/test_canonical.py`, `test_corruption.py`,
  `test_cleaner_fixtures.py`, `test_executor_verifier.py`, and new
  `test_table_semantics.py`
- Documentation: `docs/A0_AUDIT.md`, this report
- Regenerated audit outputs: `artifacts/corruption_applicability_matrix.csv`,
  `artifacts/task_smoke_tests.csv`, `artifacts/table_semantics_audit.csv`, and
  `artifacts/data_audit_summary.json`

Before applying the fixes, a recoverable server snapshot was saved at
`/hy-tmp/project-a-before-a0-correctness-audit-20261021.tgz`.

## Server-side acceptance results

Commands executed from `/root/project-a`:

```bash
conda run -n project-a python -m pytest -v
conda run -n project-a python scripts/validate_data_pack.py
conda run -n project-a python scripts/run_a0_data_audit.py
```

Actual results:

- pytest: **52 passed, 0 failed, 0 skipped** in 0.62 seconds;
- data-pack validation: **11/11 tables passed** exact semantic schema checks;
- corruption checks: **66 total, 63 execution smoke passes, 3 explicitly
  inapplicable, 0 failed**;
- task execution smoke checks: **19/19 passed**;
- Unicode replacement characters detected in source data: **145**.

## Known unresolved source limitations

These are recorded limitations, not silently repaired values:

1. `zomato_restaurants.csv` contains 145 Unicode replacement characters across
   text fields. The original characters are absent and cannot be reliably
   reconstructed.
2. In `sales_transactions.csv`, 441 `Sales_Amount` rows and 29 `Profit` rows
   differ from the simple displayed-field formulas by more than `0.011`
   (maximum absolute differences observed: 877.10 and 301.69 respectively).
   Source values remain canonical; those formulas are not used as oracle truth.
3. The semantics registry is manually curated for the current 11 tables. Any new
   table requires explicit review before it is admitted.
4. Domain-specific task fixtures must be added for the selected A1 task catalog
   before any generated trajectory is approved for training.

## Next gate

Proceed to A1 only after freezing the current A0 code, registry, and audit
artifacts. A1 should define the bounded structured task catalog and table-level
train/dev/test split, then add independently handwritten domain fixtures before
bulk synthetic trajectory generation. Do not start model download or training at
this gate.
