import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns

# ── 1. Load / Generate Dataset ────────────────────────────
# To use your own data: df = pd.read_csv("your_sales_data.csv")

np.random.seed(42)
n = 30000
regions  = ['North', 'South', 'East', 'West', 'Central']
products = ['Electronics', 'Clothing', 'Food', 'Furniture', 'Sports']

df = pd.DataFrame({
    'order_id':    range(1, n + 1),
    'date':        pd.date_range('2022-01-01', periods=n, freq='1h'),
    'region':      np.random.choice(regions, n),
    'product':     np.random.choice(products, n),
    'quantity':    np.random.randint(1, 20, n).astype(float),
    'unit_price':  np.random.uniform(10, 500, n).round(2),
    'customer_id': np.random.randint(1000, 5000, n),
    'discount':    np.random.choice([0, 0.05, 0.10, 0.15, 0.20], n),
})

# Introduce missing values for cleaning demo
df.loc[np.random.choice(df.index, 300), 'quantity']   = np.nan
df.loc[np.random.choice(df.index, 200), 'unit_price'] = np.nan

# ── 2. Data Cleaning ──────────────────────────────────────
print("=" * 50)
print("DATA CLEANING")
print("=" * 50)
print(f"Shape before cleaning: {df.shape}")
print(f"\nMissing values:\n{df.isnull().sum()}")

df['quantity'].fillna(df['quantity'].median(), inplace=True)
df['unit_price'].fillna(df['unit_price'].median(), inplace=True)
df.drop_duplicates(inplace=True)

df['date']    = pd.to_datetime(df['date'])
df['month']   = df['date'].dt.month
df['year']    = df['date'].dt.year
df['quarter'] = df['date'].dt.quarter

print(f"\nMissing values after cleaning:\n{df.isnull().sum()}")
print(f"Shape after cleaning: {df.shape}")

# ── 3. Feature Engineering ────────────────────────────────
df['revenue'] = (df['quantity'] * df['unit_price'] * (1 - df['discount'])).round(2)

# ── 4. KPI Calculations ───────────────────────────────────
print("\n" + "=" * 50)
print("KEY PERFORMANCE INDICATORS (KPIs)")
print("=" * 50)

total_revenue    = df['revenue'].sum()
total_orders     = df['order_id'].nunique()
avg_order_value  = df['revenue'].mean()
unique_customers = df['customer_id'].nunique()
top_product      = df.groupby('product')['revenue'].sum().idxmax()

print(f"Total Revenue:       ₹{total_revenue:,.2f}")
print(f"Total Orders:        {total_orders:,}")
print(f"Avg Order Value:     ₹{avg_order_value:,.2f}")
print(f"Unique Customers:    {unique_customers:,}")
print(f"Top Product:         {top_product}")

# ── 5. Underperforming Segments ───────────────────────────
revenue_by_product = df.groupby('product')['revenue'].sum().sort_values(ascending=False)
revenue_by_region  = df.groupby('region')['revenue'].sum().sort_values(ascending=False)
monthly_revenue    = df.groupby(['year', 'month'])['revenue'].sum().reset_index()
customer_orders    = df.groupby('customer_id')['revenue'].sum().sort_values(ascending=False)

underperforming = revenue_by_product.tail(3)
print("\n" + "=" * 50)
print("UNDERPERFORMING PRODUCT SEGMENTS (Bottom 3)")
print("=" * 50)
print(underperforming.round(2))

# ── 6. Visualizations ─────────────────────────────────────
fig, axes = plt.subplots(2, 2, figsize=(14, 10))
fig.suptitle('Sales Data Analysis Dashboard', fontsize=16, fontweight='bold')

# Revenue by Region
axes[0, 0].bar(revenue_by_region.index, revenue_by_region.values / 1e6,
               color=['#7F77DD', '#9FE1CB', '#F0997B', '#FAC775', '#85B7EB'])
axes[0, 0].set_title('Revenue by Region (₹M)')
axes[0, 0].set_xlabel('Region')
axes[0, 0].set_ylabel('Revenue (Millions ₹)')
axes[0, 0].tick_params(axis='x', rotation=15)

# Revenue by Product
colors_prod = ['#7F77DD' if p == top_product else '#D3D1C7' for p in revenue_by_product.index]
axes[0, 1].barh(revenue_by_product.index, revenue_by_product.values / 1e6, color=colors_prod)
axes[0, 1].set_title('Revenue by Product (₹M)')
axes[0, 1].set_xlabel('Revenue (Millions ₹)')

# Monthly Revenue Trend
monthly_pivot = monthly_revenue.pivot(index='month', columns='year', values='revenue')
for col in monthly_pivot.columns:
    axes[1, 0].plot(monthly_pivot.index, monthly_pivot[col] / 1e6, marker='o', label=str(col))
axes[1, 0].set_title('Monthly Revenue Trend')
axes[1, 0].set_xlabel('Month')
axes[1, 0].set_ylabel('Revenue (Millions ₹)')
axes[1, 0].legend()

# Customer Value Distribution
axes[1, 1].hist(customer_orders.values / 1000, bins=50, color='#7F77DD', edgecolor='white')
axes[1, 1].set_title('Customer Revenue Distribution (₹K)')
axes[1, 1].set_xlabel('Total Revenue per Customer (₹K)')
axes[1, 1].set_ylabel('Number of Customers')

plt.tight_layout()
plt.savefig('sales_dashboard.png', dpi=150, bbox_inches='tight')
plt.show()
print("\nDashboard saved as sales_dashboard.png")

# ── 7. Export Cleaned Data ────────────────────────────────
df.to_csv('sales_data_cleaned.csv', index=False)
print(" Cleaned data exported to sales_data_cleaned.csv")
