# save as make_split_tables_4x.py
import pandas as pd
from textwrap import dedent
from pathlib import Path

# === INPUT ===
CSV_PATH = "err_mean_std_by_fid_alg.csv"  # your CSV with columns:
# fid,CMAES,GA,GWO,JADE,LBFGSB,LSHADE,NLSHADE-RSP,PSO,Spiral-LSHADE,SSA,SciPyDE,jSO

# Desired order of algorithms (edit if needed)
alg_order = [
    "CMAES", "GA", "GWO", "JADE",
    "LBFGSB", "LSHADE", "NLSHADE-RSP", "PSO",
    "Spiral-LSHADE", "SSA", "SciPyDE", "jSO"
]

# Chunk size (4 algorithms per table)
chunk_size = 4

# === LOAD ===
df = pd.read_csv(CSV_PATH)

# sanity check
cols = df.columns.tolist()
missing = [a for a in alg_order if a not in cols]
if missing:
    raise ValueError(f"Missing columns in CSV: {missing}")

def fmt_num(x):
    if isinstance(x, (int, float)):
        if x == 0 or abs(x) == 0.0:
            return "0.000e+00"
        # scientific for very small or very large; compact otherwise
        return f"{x:.3e}" if (abs(x) < 1e-2 or abs(x) >= 1e3) else f"{x:.3g}"
    return str(x)

def to_latex_subtable(df_sub, caption, label):
    cols = ["fid"] + [c for c in df_sub.columns if c != "fid"]
    header = " & ".join(cols)

    lines = []
    for _, row in df_sub.iterrows():
        vals = [row[c] for c in cols]
        line = " & ".join(fmt_num(v) for v in vals) + " \\\\"
        lines.append(line)

    body = "\n".join(lines)

    tex = dedent(f"""
    \\begin{{table}}[htbp]
    \\centering
    \\caption{{{caption}}}
    \\label{{{label}}}
    \\begin{{tabular}}{{l{('r' * (len(cols) - 1))}}}
    \\toprule
    {header} \\\\
    \\midrule
    {body}
    \\bottomrule
    \\end{{tabular}}
    \\end{{table}}
    """).strip()
    return tex

# === SPLIT INTO GROUPS OF 4 ===
out_dir = Path(".")
tables = []
for i in range(0, len(alg_order), chunk_size):
    group = alg_order[i:i+chunk_size]
    part_idx = i // chunk_size + 1
    sub = df[["fid"] + group].copy()
    caption = f"Mean metric per function (Part {part_idx})"
    label = f"tab:mean_all_part{part_idx}"
    tex = to_latex_subtable(sub, caption, label)
    out_path = out_dir / f"mean_table_part{part_idx}.tex"
    out_path.write_text(tex, encoding="utf-8")
    tables.append(out_path.name)

print("✅ Wrote:", ", ".join(tables))
