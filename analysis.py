"""
Auto Insurance Claim Frequency Model — freMTPL2
Poisson GLM with exposure offset, evaluated with lift chart & Gini.
"""
import numpy as np
import pandas as pd
import statsmodels.api as sm
import statsmodels.formula.api as smf
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.model_selection import train_test_split

plt.rcParams['figure.dpi'] = 110
sns.set_style('whitegrid')

# 1. Load ----------------------------------------------------------------
df = pd.read_csv('freMTPL2freq.csv')

# Data-quality caps used in the standard actuarial literature (Wüthrich et al.)
df['ClaimNb'] = df['ClaimNb'].clip(upper=4)
df['Exposure'] = df['Exposure'].clip(upper=1)
df['VehAge']   = df['VehAge'].clip(upper=20)
df['DrivAge']  = df['DrivAge'].clip(upper=90)
df['BonusMalus'] = df['BonusMalus'].clip(upper=150)
df['Density_log'] = np.log(df['Density'])

print(f"Rows: {len(df):,}")
print(f"Total exposure (policy-years): {df['Exposure'].sum():,.0f}")
print(f"Total claims: {int(df['ClaimNb'].sum()):,}")
print(f"Portfolio frequency: {df['ClaimNb'].sum()/df['Exposure'].sum():.4f} claims/policy-year")

# 2. EDA -----------------------------------------------------------------
fig, axes = plt.subplots(2, 2, figsize=(12, 8))

# (a) Frequency by driver age
age_freq = df.groupby('DrivAge').apply(
    lambda g: g['ClaimNb'].sum()/g['Exposure'].sum(), include_groups=False
)
axes[0,0].plot(age_freq.index, age_freq.values, marker='o', ms=3, color='#1f4e79')
axes[0,0].set_title('Claim frequency by driver age')
axes[0,0].set_xlabel('Driver age'); axes[0,0].set_ylabel('Claims / exposure')

# (b) Frequency by vehicle age
va_freq = df.groupby('VehAge').apply(
    lambda g: g['ClaimNb'].sum()/g['Exposure'].sum(), include_groups=False
)
axes[0,1].plot(va_freq.index, va_freq.values, marker='o', ms=4, color='#c0504d')
axes[0,1].set_title('Claim frequency by vehicle age')
axes[0,1].set_xlabel('Vehicle age'); axes[0,1].set_ylabel('Claims / exposure')

# (c) Frequency by Bonus-Malus
bm_freq = df.groupby(pd.cut(df['BonusMalus'], bins=[49,55,65,75,90,110,150])).apply(
    lambda g: g['ClaimNb'].sum()/g['Exposure'].sum(), include_groups=False
)
axes[1,0].bar(range(len(bm_freq)), bm_freq.values, color='#7c9c4b')
axes[1,0].set_xticks(range(len(bm_freq)))
axes[1,0].set_xticklabels([str(i) for i in bm_freq.index], rotation=25, ha='right', fontsize=8)
axes[1,0].set_title('Claim frequency by Bonus-Malus band')
axes[1,0].set_ylabel('Claims / exposure')

# (d) Frequency by area
area_freq = df.groupby('Area', observed=True).apply(
    lambda g: g['ClaimNb'].sum()/g['Exposure'].sum(), include_groups=False
).sort_index()
axes[1,1].bar(area_freq.index.astype(str), area_freq.values, color='#8064a2')
axes[1,1].set_title('Claim frequency by Area code (A = rural … F = urban)')
axes[1,1].set_ylabel('Claims / exposure')

plt.tight_layout()
plt.savefig('01_eda.png', bbox_inches='tight')
plt.close()
print("Saved 01_eda.png")

# 3. Train / test --------------------------------------------------------
train, test = train_test_split(df, test_size=0.2, random_state=42)
print(f"Train: {len(train):,}   Test: {len(test):,}")

# 4. Poisson GLM ---------------------------------------------------------
# offset = log(Exposure) so we're modeling frequency, not count
formula = ("ClaimNb ~ C(Area) + VehPower + VehAge + DrivAge + BonusMalus "
           "+ C(VehBrand) + C(VehGas) + Density_log + C(Region)")

model = smf.glm(
    formula=formula,
    data=train,
    family=sm.families.Poisson(),
    offset=np.log(train['Exposure']),
).fit()

print("\n--- Model summary (top rows) ---")
print(model.summary().tables[0])

# Save top coefficients as an interpretable table
coef_tbl = (pd.DataFrame({'coef': model.params, 'p': model.pvalues,
                          'mult_effect': np.exp(model.params)})
              .sort_values('mult_effect', ascending=False))
coef_tbl.to_csv('coefficients.csv')
print(f"\nTop 5 rate-increasing effects:")
print(coef_tbl.head().to_string())
print(f"\nTop 5 rate-decreasing effects:")
print(coef_tbl.tail().to_string())

# 5. Prediction & evaluation --------------------------------------------
test = test.copy()
test['pred_freq'] = model.predict(test, offset=np.log(test['Exposure'])) / test['Exposure']
test['pred_count'] = test['pred_freq'] * test['Exposure']

# Lift chart: sort by predicted frequency, bucket into deciles
test['decile'] = pd.qcut(test['pred_freq'], 10, labels=False, duplicates='drop') + 1
lift = test.groupby('decile').apply(
    lambda g: pd.Series({
        'exposure':  g['Exposure'].sum(),
        'actual_freq': g['ClaimNb'].sum() / g['Exposure'].sum(),
        'pred_freq':   g['pred_count'].sum() / g['Exposure'].sum(),
    }), include_groups=False
)
print("\n--- Lift chart (test set) ---")
print(lift.to_string())

fig, ax = plt.subplots(figsize=(8, 5))
x = lift.index
w = 0.38
ax.bar([i-w/2 for i in x], lift['actual_freq'], width=w, label='Actual', color='#1f4e79')
ax.bar([i+w/2 for i in x], lift['pred_freq'],   width=w, label='Predicted', color='#c0504d')
ax.set_title('Lift chart — actual vs. predicted claim frequency by risk decile')
ax.set_xlabel('Predicted-risk decile (1 = safest, 10 = riskiest)')
ax.set_ylabel('Claim frequency')
ax.legend()
plt.tight_layout()
plt.savefig('02_lift_chart.png', bbox_inches='tight')
plt.close()
print("Saved 02_lift_chart.png")

# 6. Gini / concentration curve -----------------------------------------
tt = test.sort_values('pred_freq').reset_index(drop=True)
cum_exposure = tt['Exposure'].cumsum() / tt['Exposure'].sum()
cum_claims   = tt['ClaimNb'].cumsum()  / tt['ClaimNb'].sum()
gini = 1 - 2 * np.trapezoid(cum_claims, cum_exposure)
print(f"\nGini coefficient (concentration): {gini:.4f}")
# Baseline (mean-rate) for reference
baseline_freq = train['ClaimNb'].sum() / train['Exposure'].sum()
poisson_dev_base = 2*(sm.families.Poisson().loglike_obs(
    test['ClaimNb'], test['Exposure']*baseline_freq)).sum() * -1
poisson_dev_model = 2*(sm.families.Poisson().loglike_obs(
    test['ClaimNb'], test['pred_count'])).sum() * -1
print(f"Test deviance — mean-rate baseline: {poisson_dev_base:,.0f}")
print(f"Test deviance — GLM:                {poisson_dev_model:,.0f}")
print(f"Deviance improvement: {(poisson_dev_base-poisson_dev_model)/poisson_dev_base:.2%}")

fig, ax = plt.subplots(figsize=(6, 6))
ax.plot(cum_exposure, cum_claims, color='#1f4e79', lw=2, label=f'GLM  (Gini = {gini:.3f})')
ax.plot([0,1],[0,1], color='grey', linestyle='--', label='Random')
ax.set_title('Concentration curve — GLM ranks risk well when it lies above the diagonal')
ax.set_xlabel('Cumulative exposure (sorted by predicted frequency)')
ax.set_ylabel('Cumulative claims')
ax.legend()
plt.tight_layout()
plt.savefig('03_gini.png', bbox_inches='tight')
plt.close()
print("Saved 03_gini.png")

print("\nDone.")
