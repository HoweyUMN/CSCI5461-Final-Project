
"""
bundle_anova_full.py   –   one-way ANOVA per gene across cell groups

Quick test (2 000 cells):
python3 bundle_anova_full.py \
  --mtx matrix.mtx --barcodes barcodes.tsv --genes features.tsv \
  --cell_meta clustering/AMY/derived_features.csv --groupby cluster \ #this can change depending on the type (AMY,AUD,ACA) 
                                                                        Had to use lables from here for this to work
  --min_counts 1000 --min_cells_per_gene_group 10 --sample 2000 \
  --n_jobs 4 --outfile anova_test.csv
"""
from __future__ import annotations
import argparse, time, warnings
from typing import Tuple, Dict

import numpy as np
import pandas as pd
import scanpy as sc
import scipy.stats as st
from scipy.io import mmread
from statsmodels.stats.multitest import multipletests
from joblib import Parallel, delayed
from tqdm.auto import tqdm
from scipy.stats._axis_nan_policy import SmallSampleWarning
warnings.filterwarnings("ignore", category=SmallSampleWarning)

def filter_cells_and_genes(
    matrix_path: str,
    barcodes_path: str,
    features_path: str,
    cell_meta_csv: str,
    groupby: str,
    min_counts_per_cell: int,
    min_cells_per_gene_per_group: int,
) -> Tuple[np.ndarray, pd.Index, pd.DataFrame, np.ndarray, Dict[str, np.ndarray]]:

    print("reading sparse matrix …")
    X_all = mmread(matrix_path).T.tocsr()
    barcodes = pd.read_csv(barcodes_path, sep="\t", header=None,
                           names=["barcode"])["barcode"]
    feats = pd.read_csv(features_path, sep="\t", header=None,
                        names=["gene_id","gene_name","feature_type"])[["gene_id","gene_name"]]
    assert X_all.shape == (len(barcodes), len(feats)), "Matrix ≠ barcodes/genes"

    #Cell QC
    keep_cell = np.asarray(X_all.sum(axis=1)).ravel() >= min_counts_per_cell
    X1 = X_all[keep_cell]
    bar1 = barcodes[keep_cell].reset_index(drop=True)

    #Metadata & groups
    meta = pd.read_csv(cell_meta_csv)
    if "barcode" not in meta.columns:
        meta.columns = ["barcode", *meta.columns[1:]]
    meta = meta.set_index("barcode").loc[bar1]
    groups = meta[groupby].astype(str).values

    #Per‐group gene filter
    grp2idx: Dict[str, np.ndarray] = {
        lvl: np.where(groups == lvl)[0] for lvl in np.unique(groups)
    }
    keep_gene = np.ones(X1.shape[1], bool)
    for idxs in grp2idx.values():
        nz = np.asarray((X1[idxs] > 0).sum(axis=0)).ravel()
        keep_gene &= nz >= min_cells_per_gene_per_group
    X2 = X1[:, keep_gene]
    feats2 = feats.loc[keep_gene].reset_index(drop=True)

    print(f"▸ kept {X2.shape[0]} cells, {X2.shape[1]} genes "
          f"(≥{min_cells_per_gene_per_group} non-zeros/group)")
    return X2, bar1, feats2, groups, grp2idx

def cli() -> argparse.Namespace:
    p = argparse.ArgumentParser()
    p.add_argument("--mtx",       required=True)
    p.add_argument("--barcodes",  required=True)
    p.add_argument("--genes",     required=True)
    p.add_argument("--cell_meta", required=True)
    p.add_argument("--groupby",   default="cluster")
    p.add_argument("--min_counts",               type=int, default=1000)
    p.add_argument("--min_cells_per_gene_group", type=int, default=20)
    p.add_argument("--sample",      type=int, default=0,
                   help="sub-sample N cells after QC (0=all)")
    p.add_argument("--n_jobs",      type=int, default=1)
    p.add_argument("--outfile",     default="anova_out.csv")
    return p.parse_args()

def main() -> None:
    args = cli()
    t0 = time.time()

    X, barcodes, feats, groups, grp2idx = filter_cells_and_genes(
        args.mtx, args.barcodes, args.genes,
        args.cell_meta, args.groupby,
        args.min_counts,
        args.min_cells_per_gene_group
    )

    # optional subsample
    if args.sample and args.sample < X.shape[0]:
        sel = np.random.choice(X.shape[0], args.sample, replace=False)
        X = X[sel]
        barcodes = barcodes.iloc[sel].reset_index(drop=True)
        groups = groups[sel]
        grp2idx = {lvl: np.where(groups == lvl)[0] for lvl in np.unique(groups)}
        print(f"🔬 subsampled → {X.shape[0]} cells")

    # CPM + log1p
    sums = np.asarray(X.sum(axis=1)).ravel()
    X = X.multiply(1e6 / sums[:, None])
    X.data = np.log1p(X.data)
    X = X.tocsr()

    # info
    print("\n group sizes:\n",
          pd.Series({k: len(v) for k, v in grp2idx.items()}).sort_index(), "\n")
    genes = feats["gene_name"].tolist()
    print(f"ANOVA on {len(genes)} genes using {args.n_jobs} jobs")

    # pull out each gene‐vector via toarray().ravel()
    def worker(i: int):
        arrs = [
            X[idxs, i].toarray().ravel()
            for idxs in grp2idx.values()
        ]
        F, p = st.f_oneway(*arrs)
        return genes[i], len(arrs), F, p

    results = Parallel(n_jobs=args.n_jobs, verbose=5)(
        delayed(worker)(i) for i in range(len(genes))
    )

    df = pd.DataFrame(results, columns=["gene","n_groups","F","pval"])
    df["qval"] = multipletests(df["pval"], method="fdr_bh")[1]
    df.to_csv(args.outfile, index=False)

    print(f"\n✅ wrote {len(df):,} rows → {args.outfile}")
    print(f"⏱ elapsed {(time.time()-t0)/60:.1f} min")

if __name__ == "__main__":
    main()
