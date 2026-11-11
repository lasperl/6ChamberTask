# Imports
from pathlib import Path
import numpy as np
import pandas as pd
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import bambi as bmb
import arviz as az
import yaml
from scipy.stats import ttest_rel
from scipy import integrate
def z(v):
    return (v - v.mean()) / v.std()


# FUNCTIOND
def bf10_onesample(t, n, r=0.707):
    """JZS BF10 for a one-sample / paired t-test (Rouder et al. 2009), r = 0.707 like JASP"""
    v = n - 1
    null = (1 + t ** 2 / v) ** (-(v + 1) / 2)
    f = lambda g: ((1 + n * g * r ** 2) ** -0.5 * (1 + t ** 2 / ((1 + n * g * r ** 2) * v)) ** (-(v + 1) / 2)
                   * (2 * np.pi) ** -0.5 * g ** -1.5 * np.exp(-1 / (2 * g)))
    return integrate.quad(f, 0, np.inf)[0] / null

def in_t(xy):
    """True where the position is inside the T"""
    x, y = xy[:, 0], xy[:, 1]
    in_wide = (x >= 9) & (x <= 24) & (y >= 10) & (y <= 62)
    in_arm = (x >= 24) & (x <= 43) & (y >= 28) & (y <= 44)
    return in_wide | in_arm

def occ_map(xy):
    """proportion of time per mask bin"""
    H, _, _ = np.histogram2d(xy[:, 0], xy[:, 1], bins=[xe, ye])
    v = H[mask]  # keep only bins inside the nmask
    return v / v.sum()


# Parameters
bin_cm = 4
GEO_PCT = 0.5

#other
FOLDER = Path("/data07/Lina/6Chamber_LbO/Data_analysis")
ANNOT_DIR = FOLDER / "annotated"
PHASES = ["Baseline_Closed", "Observation_Neutral", "Observation_Shock", "Recall_Closed"]
RATS = sorted({"_".join(f.stem.split("_")[:2]) for f in ANNOT_DIR.glob("*.pkl")})
print(len(RATS), "rats")


# Cell 1: load positions (and remove non-tracked)
POS = {}
POS_HALF = {}
rows = []
for rat in RATS:
    for ph in PHASES:
        A = pd.read_pickle(ANNOT_DIR / f"{rat}_{ph}.pkl")
        ok = A.tracked_back.astype(bool).to_numpy()
        POS[(rat, ph)] = A[["x_back", "y_back"]].to_numpy(float)[ok]
        rows.append({"rat": rat, "phase": ph, "n_total": len(ok), "n_tracked": ok.sum(), 'removed': len(ok) - ok.sum(), 'perc_removed': (len(ok) - ok.sum()) / len(ok) * 100})

        #Split recall
        if ph == "Recall_Closed":
            xy = A[["x_back", "y_back"]].to_numpy(float)
            half = len(A) // 2
            first = np.arange(len(A)) < half
            POS_HALF[(rat, "Recall_1")] = xy[ok & first]
            POS_HALF[(rat, "Recall_2")] = xy[ok & ~first]

frames = pd.DataFrame(rows)
print(frames.head())

# frames.groupby('phase')['perc_removed'].mean()
# frames.sort_values('perc_removed', ascending=False).head()

# Cell 2: occupancy maps
all_xy = np.vstack(list(POS.values())) #all positions of every rat, every phase
print(all_xy[:, 0].min(), all_xy[:, 0].max())     # x limits
print(all_xy[:, 1].min(), all_xy[:, 1].max())

# plot shape of all positions (sanity)
fig, ax = plt.subplots(figsize=(6, 7))
ax.hist2d(all_xy[:, 0], all_xy[:, 1], bins=80, cmap="viridis", norm="log")
ax.set_aspect("equal"); ax.set_xlabel("x (cm)"); ax.set_ylabel("y (cm)")
plt.tight_layout(); plt.savefig(FOLDER / "all_positions_check.png", dpi=150); plt.close()

# remove outliers if still present somehow
x, y = all_xy[:, 0], all_xy[:, 1]
in_wide = (x >= 9) & (x <= 24) & (y >= 10) & (y <= 62)
in_arm = (x >= 24) & (x<= 43) & (y >= 28) & (y <= 44)
in_arena = in_wide | in_arm

# remove any outliers per rat
POS_IN = {key: xy[in_t(xy)] for key, xy in POS.items()}
xy_in = np.vstack(list(POS_IN.values()))
print(len(xy_in))

# plot shape of all positions inside now (sanity)
fig, ax = plt.subplots(figsize=(6, 7))
ax.hist2d(xy_in[:, 0], xy_in[:, 1], bins=80, cmap="viridis", norm="log")
ax.set_aspect("equal"); ax.set_xlabel("x (cm)"); ax.set_ylabel("y (cm)")
plt.tight_layout(); plt.savefig(FOLDER / "all_positions_in.png", dpi=150); plt.close()

# Cell 3: mask for GLM
# median position of the lines
L = {k: [] for k in ("line_shock", "line_safe", "line_middle")}
for rat in RATS:
    for ph in PHASES:
        Pz = np.load(FOLDER / "processed" / f"{rat}_{ph}.npz", allow_pickle=True)
        for k in L:
            L[k].append(np.asarray(Pz[k], float))
y_div1 = np.median([l[:, 1].mean() for l in L["line_shock"]])
y_div2 = np.median([l[:, 1].mean() for l in L["line_safe"]])
x_mid = np.median([l[:, 0].mean() for l in L["line_middle"]])
y_axis = (y_div1 + y_div2) / 2                    # centre line of the arm

left = all_xy[all_xy[:, 0] < x_mid]; right = all_xy[all_xy[:, 0] >= x_mid]
spread = lambda a: np.subtract(*np.percentile(a[:, 1], [100 - GEO_PCT, GEO_PCT]))
narrow_is_right = spread(right) < spread(left)
wide_pts = left if narrow_is_right else right
narrow_pts = right if narrow_is_right else left

# widths and ends of the T
half_narrow = abs(y_div2 - y_div1) / 2
lo, hi = np.percentile(wide_pts[:, 1], [GEO_PCT, 100 - GEO_PCT])
half_wide = ((y_axis - lo) + (hi - y_axis)) / 2
if narrow_is_right:
    x_wide_end = np.percentile(wide_pts[:, 0], GEO_PCT)
    x_narrow_end = np.percentile(narrow_pts[:, 0], 100 - GEO_PCT)
else:
    x_wide_end = np.percentile(wide_pts[:, 0], 100 - GEO_PCT)
    x_narrow_end = np.percentile(narrow_pts[:, 0], GEO_PCT)

# edges: grid centred on the arena
nx_lo = int(np.ceil((x_mid - min(x_wide_end, x_narrow_end)) / bin_cm)) + 1
nx_hi = int(np.ceil((max(x_wide_end, x_narrow_end) - x_mid) / bin_cm)) + 1
xe = x_mid + bin_cm * np.arange(-nx_lo, nx_hi + 1)
ny = int(np.ceil(half_wide / bin_cm)) + 1
ye = y_axis + bin_cm * np.arange(-ny, ny + 1)
print(f"grid {len(xe) - 1} x {len(ye) - 1} bins of {bin_cm} cm")


xc = (xe[:-1] + xe[1:]) / 2
yc = (ye[:-1] + ye[1:]) / 2
XC, YC = np.meshgrid(xc, yc, indexing="ij")
in_x_wide = (XC <= x_mid) & (XC >= x_wide_end) if narrow_is_right else (XC >= x_mid) & (XC <= x_wide_end)
in_x_narrow = (XC > x_mid) & (XC <= x_narrow_end) if narrow_is_right else (XC < x_mid) & (XC >= x_narrow_end)
mask = (in_x_wide & (np.abs(YC - y_axis) <= half_wide)) | (in_x_narrow & (np.abs(YC - y_axis) <= half_narrow))
print(mask.shape, mask.sum(), "bins kept after masking")


MAPS = {key: occ_map(xy) for key, xy in POS.items()}
print(len(MAPS), "maps")

# maps of the two recall halves (first half vs last half)
for key, xy in POS_HALF.items():
    MAPS[key] = occ_map(xy)
print(MAPS[(RATS[0], "Recall_1")].sum(), MAPS[(RATS[0], "Recall_2")].sum())

# map = MAPS[("F_B1C1R1", "Recall_Closed")]

keys = list(MAPS.keys())
key = keys[np.random.randint(len(keys))]     # random rat and phase
print(key)
img = np.full(mask.shape, np.nan)
img[mask] = MAPS[key]


# Cell 4: OLS for one rat (for partial correlwtions)
predictors = {"base": "Baseline_Closed", "neut": "Observation_Neutral", "shock": "Observation_Shock"}

print('OLS per rat')
order = {}
rows = []
for rec in ("Recall_Closed", "Recall_1", "Recall_2"):
    for rat in RATS:
        y = z(MAPS[(rat, rec)]) #
        X = np.column_stack([z(MAPS[(rat, ph)]) for ph in predictors.values()])
        b, *_ = np.linalg.lstsq(X, y, rcond=None)
        res = y - X @ b
        rows.append({"recall": rec, "rat": rat, "beta_base": b[0], "beta_neut": b[1], "beta_shock": b[2],
                     "R2": 1 - (res @ res) / (y @ y)})
betas_half = pd.DataFrame(rows)
print(betas_half.groupby("recall")[["beta_base", "beta_neut", "beta_shock", "R2"]].mean().round(3))

w = betas_half.pivot(index="rat", columns="recall", values="beta_neut")
d = w["Recall_1"] - w["Recall_2"]
t_, p_ = ttest_rel(w["Recall_1"], w["Recall_2"])
print(f"neutral 1st - 2nd half: mean {d.mean():.3f}, {int((d > 0).sum())}/16 rats higher in 1st half, p = {p_:.3f}")


order = {rat: yaml.safe_load(open(FOLDER / rat / "meta_data.yaml"))["first_obs"] for rat in RATS}
#print(pd.Series(order).value_counts())


# long format df for bambi
long = pd.concat([pd.DataFrame({"rat": rat, "y": z(MAPS[(rat, "Recall_Closed")]),
                                **{k: z(MAPS[(rat, ph)]) for k, ph in predictors.items()}})
                  for rat in RATS], ignore_index=True)

long["order_c"] = long.rat.map(order).map({"safe": -0.5, "shock": 0.5})
for k in predictors:
    long[f"{k}_x_order"] = long[k] * long.order_c
print(long.head())


# Cell 5: bambi OLS
m = bmb.Model("y ~ 0 + base + neut + shock + (0 + base + neut + shock | rat)", long, family="gaussian")
idata = m.fit(draws=1000, tune=1000, chains=4, target_accept=0.9, random_seed=1)

print(az.summary(idata, var_names=["base", "neut", "shock", "sigma"]))



# Cell 6: bambi with order
m_ord = bmb.Model("y ~ 0 + base + neut + shock + order_c + base_x_order + neut_x_order + shock_x_order"
                  " + (0 + base + neut + shock | rat)", long, family="gaussian")
idata_ord = m_ord.fit(draws=1000, tune=1000, chains=4, target_accept=0.9, random_seed=1)

names = ["base", "neut", "shock", "order_c", "base_x_order", "neut_x_order", "shock_x_order", "sigma"]
print(az.summary(idata_ord, var_names=names))


# Cell 7: bambi betas per rat
post = idata.posterior
rows = []
for k in predictors:
    full = post[k] + post[f"{k}|rat"]                         # group slope + rat offset
    for rat in full.coords["rat__factor_dim"].values:
        v = full.sel(rat__factor_dim=rat).values.ravel()
        lo, hi = np.percentile(v, [2.5, 97.5])
        rows.append({"rat": rat, "predictor": k, "beta": v.mean(), "ci_lo": lo, "ci_hi": hi})
rat_betas = pd.DataFrame(rows)

print(rat_betas.pivot(index="rat", columns="predictor", values="beta").round(3)[list(predictors)])
#compare = rat_betas.pivot(index="rat", columns="predictor", values="beta").join(betas.set_index("rat"))
#print(compare.round(3).to_string())




# adding sex
long["sex_c"] = long.rat.str[0].map({"F": -0.5, "M": 0.5})
for k in predictors:
    long[f"{k}_x_sex"] = long[k] * long.sex_c
print(long.groupby("sex_c").rat.nunique())

long_x = long[long.rat != "M_B2C1R1"]

# Cell 13: bambi with order and sex
m_sex = bmb.Model("y ~ 0 + base + neut + shock + order_c + base_x_order + neut_x_order + shock_x_order"
                  " + sex_c + base_x_sex + neut_x_sex + shock_x_sex"
                  " + (0 + base + neut + shock | rat)", long, family="gaussian")
idata_sex = m_sex.fit(draws=1000, tune=1000, chains=4, target_accept=0.95, random_seed=1)

names = ["base", "neut", "shock", "base_x_order", "neut_x_order", "shock_x_order",
         "sex_c", "base_x_sex", "neut_x_sex", "shock_x_sex", "sigma"]
print(az.summary(idata_sex, var_names=names))


post = idata_sex.posterior
for v in ["base", "neut", "shock", "base_x_sex", "neut_x_sex", "shock_x_sex",
          "base_x_order", "neut_x_order", "shock_x_order"]:
    d = post[v].values.ravel()
    lo, hi = np.percentile(d, [2.5, 97.5])
    print(f"{v:14s} {d.mean():+.3f}  95% CrI [{lo:+.3f}, {hi:+.3f}]  P(> 0) = {(d > 0).mean():.3f}")



# betas per rat from the sex + order model, and save
post = idata_sex.posterior
codes = long.groupby("rat")[["order_c", "sex_c"]].first()           # each rat's order and sex code
rows = []
for k in predictors:
    for rat in post[f"{k}|rat"].coords["rat__factor_dim"].values:
        v = (post[k] + codes.loc[rat, "order_c"] * post[f"{k}_x_order"]
             + codes.loc[rat, "sex_c"] * post[f"{k}_x_sex"]
             + post[f"{k}|rat"].sel(rat__factor_dim=rat)).values.ravel()
        lo, hi = np.percentile(v, [2.5, 97.5])
        rows.append({"rat": rat, "predictor": k, "beta": v.mean(), "ci_lo": lo, "ci_hi": hi})
sex_betas = pd.DataFrame(rows)
out = sex_betas.pivot(index="rat", columns="predictor", values="beta")[list(predictors)].add_prefix("beta_")
out.to_csv(FOLDER / "glm_betas_per_rat.csv")
sex_betas.to_csv(FOLDER / "glm_betas_per_rat_with_ci.csv", index=False)
print(out.round(3).to_string())


# group results of this model
az.summary(idata_sex, var_names=names).to_csv(FOLDER / "glm_sex_order_summary.csv")



# GLM figure - betas per rat + group beta only
NAMES = {"base": "Baseline", "neut": "Obs neutral", "shock": "Obs shock"}
SEX_COL = {"F": "tab:orange", "M": "tab:green"}
ORD_MK = {"safe": "o", "shock": "s"}
rng_j = np.random.default_rng(0)

fig, ax = plt.subplots(figsize=(6, 4.8))
rb = rat_betas.pivot(index="rat", columns="predictor", values="beta")
for i, k in enumerate(predictors):
    for rat, v in rb[k].items():                                   # one dot per rat
        ax.scatter(i + rng_j.uniform(-0.15, 0.15), v, color=SEX_COL[rat[0]], marker=ORD_MK[order[rat]],
                   alpha=0.6, s=28, zorder=2)
    g = idata.posterior[k].values.ravel()                          # group beta
    lo, hi = np.percentile(g, [2.5, 97.5])
    ax.errorbar(i + 0.3, g.mean(), yerr=[[g.mean() - lo], [hi - g.mean()]], fmt="D", color="k",
                ms=8, capsize=4, lw=2, zorder=3)
ax.axhline(0, color="grey", lw=0.8, ls="--")
ax.set_xticks(range(len(predictors))); ax.set_xticklabels([NAMES[k] for k in predictors])
ax.set_ylabel("beta (recall ~ map)")
ax.set_title("recall map ~ baseline + neutral + shock\ndots = rats, black = group [95% CrI]", fontsize=10)
h = [plt.Line2D([], [], color=SEX_COL[s], marker="o", ls="", label=s) for s in SEX_COL] + \
    [plt.Line2D([], [], color="0.4", marker=ORD_MK[o], ls="", label=f"{o} first") for o in ORD_MK]
ax.legend(handles=h, fontsize=8, frameon=False, ncol=2)
ax.spines[["top", "right"]].set_visible(False)
plt.tight_layout(); plt.savefig(FOLDER / "glm_betas_all_rats.png", dpi=200); plt.close()



#EXCLUDE = ["M_B2C1R1"]
# Plot occupacny maps averages for males and females - neutral obs and recall (maybe also correlation between them)
SHOW = ["Baseline_Closed", "Observation_Neutral", "Observation_Shock", "Recall_Closed"]
ext = [xe[0], xe[-1], ye[0], ye[-1]]

fig, axes = plt.subplots(3, len(SHOW), figsize=(3.2 * len(SHOW), 9))
for j, ph in enumerate(SHOW):
    mean_sex = {s: np.mean([MAPS[(r, ph)] for r in RATS if r[0] == s], axis=0) for s in "FM"}
    #mean_sex = {s: np.mean([MAPS[(r, ph)] for r in RATS if r[0] == s and r not in EXCLUDE], axis=0) for s in "FM"}
    vmax = max(mean_sex["F"].max(), mean_sex["M"].max())          # same colour scale for F and M
    for i, (lab, v) in enumerate([("F", mean_sex["F"]), ("M", mean_sex["M"]), ("M - F", mean_sex["M"] - mean_sex["F"])]):
        img = np.full(mask.shape, np.nan); img[mask] = v
        if lab == "M - F":
            lim = np.nanmax(np.abs(img))
            im = axes[i, j].imshow(img.T, origin="lower", extent=ext, cmap="RdBu_r", vmin=-lim, vmax=lim)
        else:
            im = axes[i, j].imshow(img.T, origin="lower", extent=ext, cmap="viridis", vmin=0, vmax=vmax)
        plt.colorbar(im, ax=axes[i, j], fraction=0.04)
        axes[i, j].set_xticks([]); axes[i, j].set_yticks([])
    axes[0, j].set_title(ph.replace("_Closed", "").replace("Observation_", "Obs_"))
for i, lab in enumerate(["F (mean)", "M (mean)", "M - F (red = M more)"]):
    axes[i, 0].set_ylabel(lab)
for ax in axes.flat:
    ax.text(0.5, 0.01, "shock side", transform=ax.transAxes, ha="center", va="bottom", fontsize=7, color="tab:red")
    ax.text(0.5, 0.99, "safe side", transform=ax.transAxes, ha="center", va="top", fontsize=7, color="tab:blue")
#plt.tight_layout(); plt.savefig(FOLDER / "mean_maps_by_sex.png", dpi=150); plt.close()
plt.tight_layout(); plt.savefig(FOLDER / "mean_maps_by_sex_excl.png", dpi=150); plt.close()

# correlation between the maps per rat (plain r, no other maps removed)
# cr = pd.DataFrame([{"rat": r, "sex": r[0],
#                     "r_neut_recall": np.corrcoef(MAPS[(r, "Observation_Neutral")], MAPS[(r, "Recall_Closed")])[0, 1],
#                     "r_base_recall": np.corrcoef(MAPS[(r, "Baseline_Closed")], MAPS[(r, "Recall_Closed")])[0, 1],
#                     "r_base_neut": np.corrcoef(MAPS[(r, "Baseline_Closed")], MAPS[(r, "Observation_Neutral")])[0, 1]}
#                    for r in RATS])
# print(cr.round(3).to_string(index=False))
# print(cr.groupby("sex")[["r_neut_recall", "r_base_recall", "r_base_neut"]].mean().round(3))



#Plot Baseline occupancy same as above





#########################################################################################
# Build df to correlate betas with other stuff

# betas per rat from the sex + order model, overall and per-sex group betas
NAMES = {"base": "Baseline", "neut": "Obs neutral", "shock": "Obs shock"}
SEX_COL = {"F": "tab:orange", "M": "tab:green"}
ORD_MK = {"safe": "o", "shock": "s"}
rng_j = np.random.default_rng(0)
post = idata_sex.posterior
codes = long.groupby("rat")[["order_c", "sex_c"]].first()

fig, ax = plt.subplots(figsize=(7, 4.8))
for i, k in enumerate(predictors):
    # one dot per rat: main + order + sex part + own offset
    for rat in post[f"{k}|rat"].coords["rat__factor_dim"].values:
        v = (post[k] + codes.loc[rat, "order_c"] * post[f"{k}_x_order"]
             + codes.loc[rat, "sex_c"] * post[f"{k}_x_sex"]
             + post[f"{k}|rat"].sel(rat__factor_dim=rat)).values.mean()
        ax.scatter(i + rng_j.uniform(-0.15, 0.15), v, color=SEX_COL[rat[0]], marker=ORD_MK[order[rat]],
                   alpha=0.6, s=28, zorder=2)
    # group betas [95 % CrI]: overall (black), females, males
    for dx, col, code in ((0.25, "k", None), (0.38, SEX_COL["F"], -0.5), (0.51, SEX_COL["M"], 0.5)):
        g = (post[k] if code is None else post[k] + code * post[f"{k}_x_sex"]).values.ravel()
        lo, hi = np.percentile(g, [2.5, 97.5])
        ax.errorbar(i + dx, g.mean(), yerr=[[g.mean() - lo], [hi - g.mean()]], fmt="D", color=col,
                    mec="k", ms=7, capsize=3, lw=1.8, zorder=3)
ax.axhline(0, color="grey", lw=0.8, ls="--")
ax.set_xticks(range(len(predictors))); ax.set_xticklabels([NAMES[k] for k in predictors])
ax.set_ylabel("beta (recall ~ map")
ax.set_title("recall map ~ baseline + neutral + shock\n"
             "diamonds = group [95% CrI]: black = all, orange = F, green = M", fontsize=9)
h = [plt.Line2D([], [], color=SEX_COL[s], marker="o", ls="", label=s) for s in SEX_COL] + \
    [plt.Line2D([], [], color="0.4", marker=ORD_MK[o], ls="", label=f"{o} first") for o in ORD_MK]
ax.legend(handles=h, fontsize=8, frameon=False, ncol=2)
ax.spines[["top", "right"]].set_visible(False)
plt.tight_layout(); plt.savefig(FOLDER / "glm_betas_sex_model.png", dpi=200); plt.close()

