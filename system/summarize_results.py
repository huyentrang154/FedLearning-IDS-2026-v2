    if not matches:
        print(f"[SKIP] No Best performance metrics block: {log_path}")
        return None

    start = matches[-1].end()
    remaining = text[start:]
    endings = [
        m.start() for pat in (
            r"\n\s*-{5,}\s*Round\s+number\s*:",
            r"\n\s*Round\s+number\s*:",
            r"\n\s*Best\s+performance\s+metrics",
        )
        for m in re.finditer(pat, remaining, re.I)
    ]
    block = remaining[:min(endings)] if endings else remaining

    values = {}
    for metric in METRICS:
        pattern = (
            r"^\s*" + re.escape(metric)
            + r"\s*:\s*([-+]?(?:\d+(?:\.\d*)?|\.\d+)(?:[eE][-+]?\d+)?)"
        )
        value = search_value(pattern, block)
        values[metric] = float(value) if value is not None else None

    if all(v is None for v in values.values()):
        print(f"[SKIP] Metric labels not recognized in {log_path}")
        return None

    return {
        "dataset": dataset,
        "data_type": data_type,
        "algorithm": algorithm,
        "exp_name": exp_name,
        "seed": seed,
        "noise_type": noise_type,
        "noise_ratio": noise_ratio,
        "noise_parameter": noise_parameter,
        **values,
        "log_file": str(log_path),
    }


def main():
    if not RESULTS_DIR.is_dir():
        raise SystemExit("Không tìm thấy thư mục results/. Hãy chạy script tại thư mục gốc repo.")

    OUTPUT_DIR.mkdir(parents=True, exist_ok=True)
    records = []

    for log_path in RESULTS_DIR.rglob("tracking.log"):
        if OUTPUT_DIR in log_path.parents:
            continue
        try:
            record = parse_log(log_path)
        except OSError as exc:
            print(f"[ERROR] {log_path}: {exc}")
            continue
        if record is None:
            continue
        if record["seed"] not in VALID_SEEDS:
            print(f"[SKIP] Seed không thuộc 0, 1, 2: {log_path} (seed={record['seed']})")
            continue
        records.append(record)

    if not records:
        raise SystemExit(
            "Không tìm thấy log hợp lệ. Hãy kiểm tra cấu trúc thư mục, seed "
            "và tên các metric trong tracking.log."
        )

    details = pd.DataFrame(records)
    group_cols = [
        "dataset", "data_type", "algorithm",
        "noise_type", "noise_ratio", "noise_parameter",
    ]

    # Warn about duplicate experiment runs for the same configuration and seed.
    duplicate_cols = group_cols + ["seed"]
    duplicate_mask = details.duplicated(duplicate_cols, keep=False)
    if duplicate_mask.any():
        print(
            "\n[CẢNH BÁO] Có nhiều log cùng cấu hình và seed. "
            "Các log đó đều được tính vào tổng hợp; hãy kiểm tra exp_name:"
        )
        print(details.loc[duplicate_mask, duplicate_cols + ["exp_name", "log_file"]].to_string(index=False))

    detail_columns = [
        "dataset", "data_type", "algorithm", "exp_name", "seed",
        "noise_type", "noise_ratio", "noise_parameter",
        *METRICS, "log_file",
    ]
    details = details.reindex(columns=detail_columns)
    details.to_csv(OUTPUT_DIR / "summary_per_seed.csv", index=False, encoding="utf-8-sig")

    summary_rows = []
    for keys, group in details.groupby(group_cols, dropna=False):
        if not isinstance(keys, tuple):
            keys = (keys,)
        row = dict(zip(group_cols, keys))
        row["num_logs"] = len(group)
        row["seeds_present"] = ",".join(
            str(int(s)) for s in sorted(group["seed"].dropna().unique())
        )
        for metric in METRICS:
            vals = pd.to_numeric(group[metric], errors="coerce").dropna()
            row[f"{metric}_mean"] = vals.mean() if len(vals) else None
            row[f"{metric}_std"] = vals.std(ddof=1) if len(vals) >= 2 else None
            row[f"{metric}_count"] = int(vals.count())
        summary_rows.append(row)

    summary = pd.DataFrame(summary_rows)
    summary.to_csv(OUTPUT_DIR / "summary_mean_std.csv", index=False, encoding="utf-8-sig")

    excel_path = OUTPUT_DIR / "summary_mean_std.xlsx"
    with pd.ExcelWriter(excel_path, engine="openpyxl") as writer:
        summary.to_excel(writer, sheet_name="Mean_Std", index=False)
        details.to_excel(writer, sheet_name="Per_Log", index=False)

    print(f"\nĐã đọc {len(details)} log hợp lệ, chỉ sử dụng seed_0, seed_1, seed_2.")
    print(f"Chi tiết từng log: {OUTPUT_DIR / 'summary_per_seed.csv'}")
    print(f"Tổng hợp CSV:      {OUTPUT_DIR / 'summary_mean_std.csv'}")
    print(f"Tổng hợp Excel:    {excel_path}")


if __name__ == "__main__":
    main()
