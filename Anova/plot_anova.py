
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from scipy.stats import f
from statsmodels.stats.multitest import multipletests

#configurations for this 
CSV = "region_AUD.csv"   #anova output
PV_COL = "pval"
F_COL  = "F"
CAP_PCTL = 98              
ALPHA = 0.05                
#load data
df = pd.read_csv(CSV)
# replace zeros so -log10 works
p = df[PV_COL].replace(0, 1e-300)
logp = -np.log10(p)

# for multiple‐testing correction
df["qval2"] = multipletests(df[PV_COL], method="fdr_bh")[1]

# histogram of p-values 
plt.figure(figsize=(6,4))
plt.hist(df[PV_COL], bins=50, edgecolor="gray")
plt.xlabel("raw p-value")
plt.ylabel("count")
plt.title("P-value distribution")
plt.tight_layout()
plt.savefig("anova_pval_hist.png", dpi=150)
plt.show()

# Q–Q plot 
theory_q = np.linspace(0,1,len(p), endpoint=False)[1:]  # skip 0
emp_q   = np.sort(p.values)[1:]                       # skip the smallest
# make them same length
n = min(len(theory_q), len(emp_q))
theory_q, emp_q = theory_q[:n], emp_q[:n]

x = -np.log10(theory_q)
y = -np.log10(emp_q)

cap = np.percentile(y, CAP_PCTL)
x_cap = np.percentile(x, CAP_PCTL)

plt.figure(figsize=(5,5))
plt.plot(x, y, ".", ms=3, alpha=0.6)
plt.plot([0, x_cap],[0, x_cap], "k--", lw=1)
plt.xlim(0, x_cap)
plt.ylim(0, cap)
plt.xlabel("theoretical –log10(p)")
plt.ylabel("empirical  –log10(p)")
plt.title("Q–Q plot of ANOVA p-values")
plt.tight_layout()
plt.savefig("anova_qq.png", dpi=150)
plt.show()

# Volcano-style: F vs –log10(p)
F = df[F_COL].values
plt.figure(figsize=(6,5))
plt.scatter(F, logp, s=5, alpha=0.5)
# highlight FDR-significant
sig = df["qval2"] < ALPHA
plt.scatter(F[sig], logp[sig], s=6, alpha=0.8, color="red", label=f"q<{ALPHA}")
plt.xlabel("F-statistic")
plt.ylabel("–log10(p-value)")
plt.title("ANOVA volcano plot")
# cap axes
plt.xlim(0, np.percentile(F, CAP_PCTL))
plt.ylim(0, np.percentile(logp, CAP_PCTL))
plt.legend(markerscale=2)
plt.tight_layout()
plt.savefig("anova_volcano.png", dpi=150)
plt.show()
