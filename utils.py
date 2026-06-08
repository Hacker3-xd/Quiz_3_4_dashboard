import os
import pandas as pd

RENAME_MAP = {
    "Gross Domestic Product (GDP)": "GDP",
    "Gross National Income(GNI) in USD": "GNI",
    "Population": "Population",
    "AMA exchange rate": "Exchange_Rate",
    "IMF based exchange rate": "IMF_Exchange_Rate",
    "Per capita GNI": "Per_Capita_GNI",
    "Exports of goods and services": "Exports",
    "Imports of goods and services": "Imports",
    "Agriculture, hunting, forestry, fishing (ISIC A-B)": "Agriculture",
    "Manufacturing (ISIC D)": "Manufacturing",
    "Construction (ISIC F)": "Construction",
    "Final consumption expenditure": "Final_Consumption",
    "Gross capital formation": "Gross_Capital_Formation",
    "General government final consumption expenditure": "Gov_Consumption",
    "Household consumption expenditure (including Non-profit institutions serving households)": "Household_Consumption",
    "Changes in inventories": "Inventories",
    "Total Value Added": "Total_Value_Added",
    "Transport, storage and communication (ISIC I)": "Transport",
    "Wholesale, retail trade, restaurants and hotels (ISIC G-H)": "Wholesale_Retail",
    "Mining, Manufacturing, Utilities (ISIC C-E)": "Mining_Manufacturing",
    "Other Activities (ISIC J-P)": "Other_Activities",
    "Gross fixed capital formation (including Acquisitions less disposals of valuables)": "Gross_Fixed_Capital",
}

REGION_MAP = {
    "United States": "North America", "Canada": "North America",
    "Mexico": "Latin America", "Brazil": "Latin America",
    "Argentina": "Latin America", "Colombia": "Latin America",
    "Chile": "Latin America", "Peru": "Latin America",
    "Venezuela": "Latin America",
    "China": "East Asia", "Japan": "East Asia",
    "Korea Rep": "East Asia", "Taiwan": "East Asia",
    "Hong Kong": "East Asia",
    "India": "South Asia", "Pakistan": "South Asia",
    "Bangladesh": "South Asia", "Sri Lanka": "South Asia",
    "Germany": "Europe", "France": "Europe",
    "United Kingdom": "Europe", "Italy": "Europe",
    "Spain": "Europe", "Netherlands": "Europe",
    "Sweden": "Europe", "Poland": "Europe",
    "Belgium": "Europe", "Austria": "Europe",
    "Switzerland": "Europe", "Norway": "Europe",
    "Denmark": "Europe", "Finland": "Europe",
    "Russia": "Europe & Central Asia",
    "Ukraine": "Europe & Central Asia",
    "Kazakhstan": "Europe & Central Asia",
    "Saudi Arabia": "Middle East", "Turkey": "Middle East",
    "Iran": "Middle East", "Iraq": "Middle East",
    "Israel": "Middle East", "Qatar": "Middle East",
    "Kuwait": "Middle East", "UAE": "Middle East",
    "Nigeria": "Sub-Saharan Africa",
    "South Africa": "Sub-Saharan Africa",
    "Kenya": "Sub-Saharan Africa",
    "Ethiopia": "Sub-Saharan Africa",
    "Ghana": "Sub-Saharan Africa",
    "Tanzania": "Sub-Saharan Africa",
    "Egypt Arab Rep": "Middle East & North Africa",
    "Morocco": "Middle East & North Africa",
    "Algeria": "Middle East & North Africa",
    "Tunisia": "Middle East & North Africa",
    "Australia": "East Asia & Pacific",
    "Indonesia": "East Asia & Pacific",
    "Thailand": "East Asia & Pacific",
    "Malaysia": "East Asia & Pacific",
    "Philippines": "East Asia & Pacific",
    "Vietnam": "East Asia & Pacific",
    "Singapore": "East Asia & Pacific",
    "New Zealand": "East Asia & Pacific",
}


def load_economy(path: str = "data/global_economy.csv") -> pd.DataFrame:
    """
    Safely load global_economy.csv which has duplicate 'Year' column.
    Reads with header=None to manually deduplicate headers before
    pandas ever sees them — this is the ONLY reliable fix.
    """
    if not os.path.exists(path):
        raise FileNotFoundError(f"Dataset not found: {path}")

    raw = pd.read_csv(path, header=None, low_memory=False)

    headers = raw.iloc[0].astype(str).tolist()
    df = raw.iloc[1:].reset_index(drop=True)

    seen = {}
    final_headers = []
    for col in headers:
        col = col.strip()
        if col in seen:
            seen[col] += 1
            final_headers.append(f"__DROP_{col}_{seen[col]}__")
        else:
            seen[col] = 0
            final_headers.append(col)

    df.columns = final_headers

    drop_cols = [c for c in df.columns if c.startswith("__DROP_")]
    df = df.drop(columns=drop_cols)

    df.rename(columns={k: v for k, v in RENAME_MAP.items() if k in df.columns}, inplace=True)

    TEXT_COLS = {"Country", "Currency", "CountryID"}
    for col in df.columns:
        if col not in TEXT_COLS:
            df[col] = pd.to_numeric(df[col], errors="coerce")

    if "Region" not in df.columns:
        df["Region"] = df["Country"].map(REGION_MAP).fillna("Other")

    df = df.dropna(subset=["Country"]).reset_index(drop=True)

    return df
