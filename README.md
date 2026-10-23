# E-Commerce Customer RFM Segmentation

This pipeline builds Recency, Frequency, and Monetary features from transaction data, assigns within-dataset RFM scores, and applies scaled K-Means clustering. It uses the transaction rows you provide; it does not generate sample customers or claim universal persona labels.

## Input format

Provide a CSV with one row per transaction or transaction line. Required columns default to customer_id, order_date, and amount. A distinct order identifier is recommended; common names such as order_id, invoice_no, invoice_number, and transaction_id are detected automatically. If no order identifier is available, frequency counts valid rows.

Amounts are summed as supplied. Negative refunds reduce monetary totals, but their rows still contribute to frequency when they have a valid order ID. Clean or aggregate refunds upstream if your business defines frequency differently.

## Run locally

Python 3.10 or newer is recommended.

~~~sh
python -m venv .venv
. .venv/bin/activate
python -m pip install -r requirements.txt
python rfm_segmentation.py --input transactions.csv --output customer_segments.csv --summary-output summary.json
~~~

On Windows, activate with .venv\Scripts\Activate.ps1.

Use --customer-column, --date-column, and --amount-column to select different headers. Set --order-column when the order ID column has a custom name. --as-of accepts YYYY-MM-DD; without it, the reference date is one day after the latest valid transaction. Choose --clusters to set the K-Means cluster count.

The customer CSV contains recency days, order frequency, monetary total, 1–5 R/F/M scores, a combined score, a heuristic segment, and an arbitrary K-Means cluster ID. The JSON summary reports skipped rows and average profiles for each cluster.

## Docker

~~~sh
docker build -t customer-rfm .
docker run --rm -v "$PWD:/data" customer-rfm --input /data/transactions.csv --output /data/customer_segments.csv --summary-output /data/summary.json
~~~

Cluster labels have no natural ordering and can change when the input changes. RFM segment names are transparent heuristics, not validated personas. This is descriptive segmentation, not a causal model or a guarantee of future customer behavior.