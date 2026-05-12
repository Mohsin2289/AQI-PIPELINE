import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import matplotlib.gridspec as gridspec
import seaborn as sns
import os

# =========================
# CONFIG
# =========================
CLEAN_FILE = "karachi_clean_dataset.csv"
OUTPUT_DIR = "eda_plots"
os.makedirs(OUTPUT_DIR, exist_ok=True)

plt.style.use('dark_background')
COLORS = ['#00ff88', '#00cfff', '#ff9800', '#f44336', '#9c27b0']

print("Loading dataset...")
df = pd.read_csv(CLEAN_FILE)
df.columns = df.columns.str.strip().str.lower()
df["timestamp"] = pd.to_datetime(df["timestamp"])
print(f"Total rows: {len(df)}")
print(f"Date range: {df['timestamp'].min()} to {df['timestamp'].max()}")

AQI_LABELS = {1: "Good", 2: "Fair", 3: "Moderate", 4: "Poor", 5: "Very Poor"}
AQI_COLORS = {1: '#00ff88', 2: '#fff176', 3: '#ff9800', 4: '#f44336', 5: '#9c27b0'}

# =========================
# PLOT 1 — AQI Distribution
# =========================
fig, ax = plt.subplots(figsize=(10, 5))
fig.patch.set_facecolor('#0d1117')
ax.set_facecolor('#0d1117')

counts = df['aqi'].value_counts().sort_index()
bars = ax.bar(
    [AQI_LABELS[i] for i in counts.index],
    counts.values,
    color=[AQI_COLORS[i] for i in counts.index],
    edgecolor='none', width=0.6
)

for bar, val in zip(bars, counts.values):
    ax.text(bar.get_x() + bar.get_width()/2, bar.get_height() + 20,
            str(val), ha='center', va='bottom', color='white', fontsize=11, fontweight='bold')

ax.set_title("AQI Level Distribution — Karachi (Nov 2025 – May 2026)",
             color='white', fontsize=14, fontweight='bold', pad=15)
ax.set_xlabel("AQI Level", color='#4a5568', fontsize=11)
ax.set_ylabel("Count", color='#4a5568', fontsize=11)
ax.tick_params(colors='white')
for sp in ax.spines.values():
    sp.set_visible(False)
ax.grid(axis='y', alpha=0.1, color='white')
plt.tight_layout()
plt.savefig(f"{OUTPUT_DIR}/01_aqi_distribution.png", dpi=150, bbox_inches='tight',
            facecolor='#0d1117')
plt.close()
print("✅ Plot 1: AQI Distribution saved")

# =========================
# PLOT 2 — AQI Over Time
# =========================
fig, ax = plt.subplots(figsize=(14, 5))
fig.patch.set_facecolor('#0d1117')
ax.set_facecolor('#0d1117')

df_sorted = df.sort_values("timestamp")
aqi_vals  = df_sorted["aqi"].tolist()
ts_vals   = df_sorted["timestamp"].tolist()

for i in range(len(ts_vals) - 1):
    clr = AQI_COLORS.get(aqi_vals[i], '#888')
    ax.plot(ts_vals[i:i+2], aqi_vals[i:i+2], color=clr, linewidth=0.8, alpha=0.9)

ax.fill_between(ts_vals, aqi_vals, alpha=0.05, color='#00cfff')
ax.set_yticks([1, 2, 3, 4, 5])
ax.set_yticklabels(["Good", "Fair", "Moderate", "Poor", "Very Poor"],
                   color='white', fontsize=9)
ax.set_xlabel("Date", color='#4a5568', fontsize=11)
ax.set_title("AQI Trend Over Time — Karachi",
             color='white', fontsize=14, fontweight='bold', pad=15)
ax.tick_params(axis='x', colors='#aaaaaa', labelsize=8)
for sp in ax.spines.values():
    sp.set_visible(False)
ax.grid(alpha=0.05, color='white')
plt.tight_layout()
plt.savefig(f"{OUTPUT_DIR}/02_aqi_over_time.png", dpi=150, bbox_inches='tight',
            facecolor='#0d1117')
plt.close()
print("✅ Plot 2: AQI Over Time saved")

# =========================
# PLOT 3 — Monthly AQI Trend
# =========================
fig, ax = plt.subplots(figsize=(10, 5))
fig.patch.set_facecolor('#0d1117')
ax.set_facecolor('#0d1117')

monthly = df.groupby("month")["aqi"].mean()
month_names = {11:"Nov", 12:"Dec", 1:"Jan", 2:"Feb", 3:"Mar", 4:"Apr", 5:"May"}
x = [month_names.get(m, str(m)) for m in monthly.index]

bars = ax.bar(x, monthly.values, color=COLORS[:len(x)], edgecolor='none', width=0.6)
for bar, val in zip(bars, monthly.values):
    ax.text(bar.get_x() + bar.get_width()/2, bar.get_height() + 0.02,
            f'{val:.2f}', ha='center', va='bottom', color='white', fontsize=10)

ax.set_title("Average AQI by Month", color='white', fontsize=14, fontweight='bold', pad=15)
ax.set_xlabel("Month", color='#4a5568', fontsize=11)
ax.set_ylabel("Average AQI", color='#4a5568', fontsize=11)
ax.tick_params(colors='white')
for sp in ax.spines.values():
    sp.set_visible(False)
ax.grid(axis='y', alpha=0.1, color='white')
plt.tight_layout()
plt.savefig(f"{OUTPUT_DIR}/03_monthly_aqi.png", dpi=150, bbox_inches='tight',
            facecolor='#0d1117')
plt.close()
print("✅ Plot 3: Monthly AQI saved")

# =========================
# PLOT 4 — Hourly AQI Pattern
# =========================
fig, ax = plt.subplots(figsize=(12, 5))
fig.patch.set_facecolor('#0d1117')
ax.set_facecolor('#0d1117')

hourly = df.groupby("hour")["aqi"].mean()
ax.plot(hourly.index, hourly.values, color='#00cfff', linewidth=2.5, marker='o',
        markersize=5, markerfacecolor='#ff9800', markeredgewidth=0)
ax.fill_between(hourly.index, hourly.values, alpha=0.1, color='#00cfff')

ax.set_title("Average AQI by Hour of Day", color='white', fontsize=14, fontweight='bold', pad=15)
ax.set_xlabel("Hour (UTC)", color='#4a5568', fontsize=11)
ax.set_ylabel("Average AQI", color='#4a5568', fontsize=11)
ax.set_xticks(range(0, 24, 2))
ax.tick_params(colors='white')
for sp in ax.spines.values():
    sp.set_visible(False)
ax.grid(alpha=0.08, color='white')
plt.tight_layout()
plt.savefig(f"{OUTPUT_DIR}/04_hourly_pattern.png", dpi=150, bbox_inches='tight',
            facecolor='#0d1117')
plt.close()
print("✅ Plot 4: Hourly Pattern saved")

# =========================
# PLOT 5 — Correlation Heatmap
# =========================
fig, ax = plt.subplots(figsize=(10, 8))
fig.patch.set_facecolor('#0d1117')
ax.set_facecolor('#0d1117')

cols = ["aqi", "pm25", "pm10", "no2", "co", "o3", "so2", "nh3",
        "aqi_lag_1h", "aqi_lag_3h", "aqi_change"]
corr = df[cols].corr()

mask = np.triu(np.ones_like(corr, dtype=bool))
cmap = sns.diverging_palette(10, 130, as_cmap=True)
sns.heatmap(corr, mask=mask, cmap=cmap, center=0, annot=True, fmt=".2f",
            annot_kws={"size": 8}, ax=ax,
            linewidths=0.5, linecolor='#1a1a2e',
            cbar_kws={"shrink": 0.8})

ax.set_title("Feature Correlation Heatmap", color='white', fontsize=14,
             fontweight='bold', pad=15)
ax.tick_params(colors='white', labelsize=9)
plt.setp(ax.get_xticklabels(), rotation=45, ha='right')
plt.tight_layout()
plt.savefig(f"{OUTPUT_DIR}/05_correlation_heatmap.png", dpi=150, bbox_inches='tight',
            facecolor='#0d1117')
plt.close()
print("✅ Plot 5: Correlation Heatmap saved")

# =========================
# PLOT 6 — PM2.5 vs AQI
# =========================
fig, ax = plt.subplots(figsize=(10, 5))
fig.patch.set_facecolor('#0d1117')
ax.set_facecolor('#0d1117')

for aqi_lvl in sorted(df["aqi"].unique()):
    subset = df[df["aqi"] == aqi_lvl]
    ax.scatter(subset["pm25"], subset["aqi"],
               color=AQI_COLORS.get(aqi_lvl, '#888'),
               alpha=0.3, s=10, label=AQI_LABELS.get(aqi_lvl))

ax.set_title("PM2.5 vs AQI Level", color='white', fontsize=14, fontweight='bold', pad=15)
ax.set_xlabel("PM2.5 (µg/m³)", color='#4a5568', fontsize=11)
ax.set_ylabel("AQI Level", color='#4a5568', fontsize=11)
ax.set_yticks([1, 2, 3, 4, 5])
ax.set_yticklabels(["Good", "Fair", "Moderate", "Poor", "Very Poor"], color='white')
ax.tick_params(axis='x', colors='white')
ax.legend(loc='upper left', facecolor='#1a1a2e', labelcolor='white', fontsize=9)
for sp in ax.spines.values():
    sp.set_visible(False)
ax.grid(alpha=0.05, color='white')
plt.tight_layout()
plt.savefig(f"{OUTPUT_DIR}/06_pm25_vs_aqi.png", dpi=150, bbox_inches='tight',
            facecolor='#0d1117')
plt.close()
print("✅ Plot 6: PM2.5 vs AQI saved")

# =========================
# PLOT 7 — Day of Week Pattern
# =========================
fig, ax = plt.subplots(figsize=(10, 5))
fig.patch.set_facecolor('#0d1117')
ax.set_facecolor('#0d1117')

days = ["Mon", "Tue", "Wed", "Thu", "Fri", "Sat", "Sun"]
weekly = df.groupby("day_of_week")["aqi"].mean()
bars = ax.bar(days, weekly.values, color='#00cfff', edgecolor='none', width=0.6, alpha=0.8)

for bar, val in zip(bars, weekly.values):
    ax.text(bar.get_x() + bar.get_width()/2, bar.get_height() + 0.01,
            f'{val:.2f}', ha='center', va='bottom', color='white', fontsize=10)

ax.set_title("Average AQI by Day of Week", color='white', fontsize=14,
             fontweight='bold', pad=15)
ax.set_xlabel("Day", color='#4a5568', fontsize=11)
ax.set_ylabel("Average AQI", color='#4a5568', fontsize=11)
ax.tick_params(colors='white')
for sp in ax.spines.values():
    sp.set_visible(False)
ax.grid(axis='y', alpha=0.1, color='white')
plt.tight_layout()
plt.savefig(f"{OUTPUT_DIR}/07_weekly_pattern.png", dpi=150, bbox_inches='tight',
            facecolor='#0d1117')
plt.close()
print("✅ Plot 7: Weekly Pattern saved")

# =========================
# SUMMARY STATS
# =========================
print("\n========== EDA SUMMARY ==========")
print(f"Total samples:     {len(df)}")
print(f"Date range:        {df['timestamp'].min().date()} to {df['timestamp'].max().date()}")
print(f"Average AQI:       {df['aqi'].mean():.2f}")
print(f"Max PM2.5:         {df['pm25'].max():.2f} µg/m³")
print(f"Hazardous hours:   {len(df[df['aqi'] >= 4])} ({len(df[df['aqi'] >= 4])/len(df)*100:.1f}%)")
print(f"Good air hours:    {len(df[df['aqi'] <= 2])} ({len(df[df['aqi'] <= 2])/len(df)*100:.1f}%)")
print(f"\nAll plots saved to: {OUTPUT_DIR}/")
print("==================================")
