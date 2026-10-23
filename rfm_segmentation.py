"""Build customer RFM features and K-Means clusters from transaction CSV data."""

from __future__ import annotations

import argparse
import json
import os
import sys
import tempfile
from datetime import date
from pathlib import Path

import numpy as np
import pandas as pd
from sklearn.cluster import KMeans
from sklearn.preprocessing import StandardScaler


class SegmentationError(ValueError):
    """Raised when transaction data cannot be segmented."""


def _resolve_column(frame: pd.DataFrame, requested: str) -> str:
    matches = [column for column in frame.columns if str(column).strip().casefold() == requested.strip().casefold()]
    if len(matches) != 1:
        raise SegmentationError(
            f"Column {requested!r} was not found exactly once. Available columns: {', '.join(map(str, frame.columns))}"
        )
    return matches[0]


def _normalize_headers(frame: pd.DataFrame) -> None:
    normalized = [str(column).strip().casefold() for column in frame.columns]
    if len(set(normalized)) != len(normalized):
        raise SegmentationError("Input has duplicate column names after trimming and case normalization.")


def _score(series: pd.Series, reverse: bool = False) -> pd.Series:
    if len(series) == 1:
        return pd.Series(3, index=series.index, dtype="int64")
    percentile = series.rank(method="average", pct=True)
    scores = np.ceil(percentile * 5).clip(1, 5).astype("int64")
    return 6 - scores if reverse else scores


def _segment(row: pd.Series) -> str:
    if row["r_score"] >= 4 and row["f_score"] >= 4 and row["m_score"] >= 4:
        return "Champions"
    if row["r_score"] <= 2 and row["f_score"] >= 3:
        return "At risk"
    if row["r_score"] >= 4 and row["f_score"] <= 2:
        return "New or promising"
    if row["f_score"] >= 4:
        return "Loyal"
    if row["r_score"] <= 2 and row["f_score"] <= 2:
        return "Hibernating"
    return "Needs attention"


def build_rfm(
    input_path: Path,
    customer_column: str = "customer_id",
    date_column: str = "order_date",
    amount_column: str = "amount",
    order_column: str | None = None,
    as_of: str | None = None,
    clusters: int = 4,
    delimiter: str = ",",
    encoding: str = "utf-8-sig",
) -> tuple[pd.DataFrame, dict]:
    source = input_path.expanduser().resolve()
    if not source.is_file():
        raise SegmentationError(f"Input does not exist or is not a file: {source}")
    if len(delimiter) != 1 or delimiter in {"\r", "\n"}:
        raise SegmentationError("Delimiter must be one non-newline character.")
    try:
        frame = pd.read_csv(source, sep=delimiter, encoding=encoding, dtype_backend="numpy_nullable")
    except TypeError:
        try:
            frame = pd.read_csv(source, sep=delimiter, encoding=encoding)
        except (OSError, UnicodeError, LookupError, pd.errors.ParserError, pd.errors.EmptyDataError) as exc:
            raise SegmentationError(f"Could not read transaction CSV: {exc}") from exc
    except (OSError, UnicodeError, LookupError, pd.errors.ParserError, pd.errors.EmptyDataError) as exc:
        raise SegmentationError(f"Could not read transaction CSV: {exc}") from exc

    if frame.empty or len(frame.columns) == 0:
        raise SegmentationError("Transaction CSV has no data rows or columns.")
    _normalize_headers(frame)
    customer_col = _resolve_column(frame, customer_column)
    date_col = _resolve_column(frame, date_column)
    amount_col = _resolve_column(frame, amount_column)
    if len({customer_col, date_col, amount_col}) < 3:
        raise SegmentationError("Customer, date, and amount columns must be different.")

    if order_column:
        order_col = _resolve_column(frame, order_column)
    else:
        order_aliases = {"order_id", "invoice_no", "invoice_number", "transaction_id"}
        found = [
            column for column in frame.columns
            if str(column).strip().casefold() in order_aliases
        ]
        if len(found) > 1:
            raise SegmentationError(
                "Multiple likely order ID columns found; pass --order-column to select one."
            )
        order_col = found[0] if found else None
    if order_col and order_col in {customer_col, date_col, amount_col}:
        raise SegmentationError("Order ID column must be different from customer, date, and amount columns.")

    customer = frame[customer_col].astype("string").str.strip()
    customer_valid = customer.notna() & customer.ne("").fillna(False)
    dates = pd.to_datetime(frame[date_col], errors="coerce", utc=True).dt.tz_convert(None)
    amounts = pd.to_numeric(frame[amount_col], errors="coerce")
    amount_valid = amounts.notna() & np.isfinite(amounts.fillna(0))
    valid_mask = customer_valid & dates.notna() & amount_valid
    if order_col:
        order_values = frame[order_col].astype("string").str.strip()
        valid_mask &= order_values.notna() & order_values.ne("").fillna(False)

    valid = pd.DataFrame(
        {
            "customer_id": customer[valid_mask].astype(str).to_numpy(),
            "order_date": dates[valid_mask].to_numpy(),
            "amount": amounts[valid_mask].astype(float).to_numpy(),
        }
    )
    if order_col:
        valid["order_id"] = order_values[valid_mask].astype(str).to_numpy()
    if valid.empty:
        raise SegmentationError("No rows have a usable customer, date, amount, and order ID (when selected).")

    latest_transaction = pd.Timestamp(valid["order_date"].max()).normalize()
    if as_of:
        reference_date = pd.to_datetime(as_of, errors="coerce", utc=True)
        if pd.isna(reference_date):
            raise SegmentationError("As-of date must be a valid date such as 2025-12-31.")
        reference_date = reference_date.tz_convert(None).normalize()
        if reference_date < latest_transaction:
            raise SegmentationError("As-of date cannot be earlier than the latest valid transaction.")
    else:
        reference_date = latest_transaction + pd.Timedelta(days=1)

    grouped = valid.groupby("customer_id", sort=True)
    rfm = grouped.agg(
        last_purchase=("order_date", "max"),
        monetary=("amount", "sum"),
    )
    if order_col:
        frequency = grouped["order_id"].nunique().rename("frequency")
    else:
        frequency = grouped.size().rename("frequency")
    rfm = rfm.join(frequency)
    rfm["recency_days"] = (reference_date - rfm["last_purchase"].dt.normalize()).dt.days.astype(int)
    rfm = rfm.reset_index()

    if not 2 <= clusters <= len(rfm):
        raise SegmentationError(f"Cluster count must be between 2 and the customer count ({len(rfm)}).")
    features = rfm[["recency_days", "frequency", "monetary"]].astype(float)
    if len(features.drop_duplicates()) < clusters:
        raise SegmentationError(
            f"Need at least {clusters} distinct RFM profiles to create {clusters} clusters."
        )

    rfm["r_score"] = _score(rfm["recency_days"], reverse=True)
    rfm["f_score"] = _score(rfm["frequency"])
    rfm["m_score"] = _score(rfm["monetary"])
    rfm["rfm_score"] = (
        rfm["r_score"].astype(str) + rfm["f_score"].astype(str) + rfm["m_score"].astype(str)
    )
    rfm["segment"] = rfm.apply(_segment, axis=1)

    scaled = StandardScaler().fit_transform(features)
    model = KMeans(n_clusters=clusters, random_state=42, n_init=10)
    rfm["cluster"] = model.fit_predict(scaled).astype(int)

    rfm["last_purchase"] = pd.to_datetime(rfm["last_purchase"]).dt.strftime("%Y-%m-%d")
    rfm = rfm[
        [
            "customer_id", "last_purchase", "recency_days", "frequency", "monetary",
            "r_score", "f_score", "m_score", "rfm_score", "segment", "cluster",
        ]
    ].sort_values(["cluster", "customer_id"], kind="stable")

    profiles = []
    for cluster_id, customers in rfm.groupby("cluster", sort=True):
        profiles.append(
            {
                "cluster": int(cluster_id),
                "customers": int(len(customers)),
                "mean_recency_days": float(customers["recency_days"].mean()),
                "mean_frequency": float(customers["frequency"].mean()),
                "mean_monetary": float(customers["monetary"].mean()),
                "segments": {
                    str(name): int(count)
                    for name, count in customers["segment"].value_counts().sort_index().items()
                },
            }
        )
    summary = {
        "source_file": source.name,
        "rows_read": int(len(frame)),
        "rows_used": int(valid_mask.sum()),
        "rows_skipped": int((~valid_mask).sum()),
        "customers": int(len(rfm)),
        "reference_date": reference_date.strftime("%Y-%m-%d"),
        "frequency_basis": "distinct order IDs" if order_col else "valid transaction rows",
        "order_column": str(order_col) if order_col else None,
        "amount_semantics": "Amounts are summed as supplied, including any negative refunds.",
        "clusters_requested": clusters,
        "cluster_profiles": profiles,
        "segment_counts": {
            str(name): int(count) for name, count in rfm["segment"].value_counts().sort_index().items()
        },
        "limitations": [
            "RFM segments use transparent heuristic rules and are not validated business personas.",
            "K-Means labels are arbitrary identifiers; compare cluster profiles, not numeric label order.",
            "The as-of date and supplied transaction coverage affect recency, frequency, and monetary values.",
        ],
    }
    return rfm, summary


def _atomic_write(path: Path, writer) -> None:
    destination = path.expanduser().resolve()
    temporary: Path | None = None
    try:
        with tempfile.NamedTemporaryFile(
            mode="w", encoding="utf-8", newline="", prefix=f".{destination.name}.",
            suffix=".tmp", dir=destination.parent, delete=False,
        ) as temp_file:
            temporary = Path(temp_file.name)
        writer(temporary)
        os.replace(temporary, destination)
        temporary = None
    except (OSError, UnicodeError, ValueError) as exc:
        raise SegmentationError(f"Could not write {destination}: {exc}") from exc
    finally:
        if temporary is not None:
            try:
                temporary.unlink(missing_ok=True)
            except OSError:
                pass


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description="Compute RFM scores and K-Means clusters from transactions.")
    parser.add_argument("--input", required=True, type=Path, help="transaction CSV")
    parser.add_argument("--output", required=True, type=Path, help="customer-level RFM CSV destination")
    parser.add_argument("--summary-output", type=Path, help="optional JSON segmentation summary")
    parser.add_argument("--customer-column", default="customer_id")
    parser.add_argument("--date-column", default="order_date")
    parser.add_argument("--amount-column", default="amount")
    parser.add_argument("--order-column", help="optional order ID column; frequency otherwise counts valid rows")
    parser.add_argument("--as-of", help="reference date in YYYY-MM-DD form; defaults to latest date plus one day")
    parser.add_argument("--clusters", type=int, default=4)
    parser.add_argument("--delimiter", default=",")
    parser.add_argument("--encoding", default="utf-8-sig")
    return parser


def main() -> int:
    parser = build_parser()
    args = parser.parse_args()
    try:
        source = args.input.expanduser().resolve()
        output = args.output.expanduser().resolve()
        summary_output = args.summary_output.expanduser().resolve() if args.summary_output else None
        if output == source or (summary_output and summary_output in {source, output}):
            raise SegmentationError("Input, CSV output, and summary output must be different files.")
        rfm, summary = build_rfm(
            args.input,
            customer_column=args.customer_column,
            date_column=args.date_column,
            amount_column=args.amount_column,
            order_column=args.order_column,
            as_of=args.as_of,
            clusters=args.clusters,
            delimiter=args.delimiter,
            encoding=args.encoding,
        )
        _atomic_write(
            args.output,
            lambda path: rfm.to_csv(path, index=False, float_format="%.2f"),
        )
        if args.summary_output:
            _atomic_write(
                args.summary_output,
                lambda path: path.write_text(
                    json.dumps(summary, indent=2, ensure_ascii=False) + "\n",
                    encoding="utf-8",
                ),
            )
    except (SegmentationError, OSError, UnicodeError, ValueError) as exc:
        print(f"Error: {exc}", file=sys.stderr)
        return 2

    print(
        f"Segmented {summary['customers']} customers from {summary['rows_used']} valid rows "
        f"({summary['rows_skipped']} skipped)."
    )
    print(f"Reference date: {summary['reference_date']} | clusters: {summary['clusters_requested']}")
    print(f"Customer CSV: {args.output.expanduser().resolve()}")
    if args.summary_output:
        print(f"Summary JSON: {args.summary_output.expanduser().resolve()}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
