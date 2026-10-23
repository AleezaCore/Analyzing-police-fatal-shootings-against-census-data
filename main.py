import pandas as pd
import plotly.express as px

# ---------- 1. Load data ----------
enc = "windows-1252"
df_hh_income = pd.read_csv("Median_Household_Income_2015.csv", encoding=enc)
df_pct_poverty = pd.read_csv("Pct_People_Below_Poverty_Level.csv", encoding=enc)
df_pct_completed_hs = pd.read_csv("Pct_Over_25_Completed_High_School.csv", encoding=enc)
df_share_race_city = pd.read_csv("Share_of_Race_By_City.csv", encoding=enc)
df_fatalities = pd.read_csv("Deaths_by_Police_US.csv", encoding=enc)

for name, df in [("income", df_hh_income), ("poverty", df_pct_poverty),
                 ("hs", df_pct_completed_hs), ("race", df_share_race_city),
                 ("fatalities", df_fatalities)]:
    print(name, df.shape, "NaN:", df.isna().sum().sum(), "Dupes:", df.duplicated().sum())

# ---------- 2. Clean ----------
df_hh_income["Median Income"] = pd.to_numeric(df_hh_income["Median Income"], errors="coerce")
df_pct_poverty["poverty_rate"] = pd.to_numeric(df_pct_poverty["poverty_rate"], errors="coerce")
df_pct_completed_hs["percent_completed_hs"] = pd.to_numeric(
    df_pct_completed_hs["percent_completed_hs"], errors="coerce")

race_cols = ["share_white", "share_black", "share_native_american",
             "share_asian", "share_hispanic"]
for c in race_cols:
    df_share_race_city[c] = pd.to_numeric(df_share_race_city[c], errors="coerce")

df_hh_income = df_hh_income.dropna()
df_pct_poverty = df_pct_poverty.dropna()
df_pct_completed_hs = df_pct_completed_hs.dropna()
df_share_race_city = df_share_race_city.dropna()

# ---------- 3. Poverty rate by state ----------
poverty_state = (df_pct_poverty.groupby("Geographic Area")["poverty_rate"]
                 .mean().sort_values(ascending=False).reset_index())
fig = px.bar(poverty_state, x="Geographic Area", y="poverty_rate",
             title="Poverty Rate by US State")
fig.show()

# ---------- 4. High school graduation rate by state ----------
hs_state = (df_pct_completed_hs.groupby("Geographic Area")["percent_completed_hs"]
            .mean().sort_values().reset_index())
fig = px.bar(hs_state, x="Geographic Area", y="percent_completed_hs",
             title="High School Graduation Rate by State")
fig.show()

# ---------- 5. Poverty vs graduation rate ----------
merged = poverty_state.merge(hs_state, on="Geographic Area")
fig = px.scatter(merged, x="poverty_rate", y="percent_completed_hs",
                 hover_name="Geographic Area", trendline="ols",
                 title="Poverty vs High School Graduation Rate")
fig.show()

# ---------- 6. Racial makeup by state ----------
race_state = df_share_race_city.groupby("Geographic area")[race_cols].mean().reset_index()
fig = px.bar(race_state, x="Geographic area", y=race_cols, barmode="stack",
             title="Racial Makeup by State (%)")
fig.show()

# ---------- 7. Deaths by race ----------
race_counts = df_fatalities["race"].value_counts().reset_index()
race_counts.columns = ["race", "count"]
fig = px.pie(race_counts, names="race", values="count", title="Police Killings by Race")
fig.show()

# ---------- 8. Gender, age, armed status ----------
print(df_fatalities["gender"].value_counts(normalize=True) * 100)
fig = px.histogram(df_fatalities, x="age", nbins=40, title="Age of People Killed")
fig.show()
armed = df_fatalities["armed"].fillna("unknown").value_counts().head(10).reset_index()
armed.columns = ["armed", "count"]
fig = px.bar(armed, x="armed", y="count", title="Top 10 'Armed' Categories")
fig.show()

# ---------- 9. Mental illness ----------
fig = px.pie(df_fatalities, names="signs_of_mental_illness",
             title="Signs of Mental Illness")
fig.show()

# ---------- 10. Top cities ----------
top_cities = df_fatalities["city"].value_counts().head(10).reset_index()
top_cities.columns = ["city", "count"]
fig = px.bar(top_cities, x="city", y="count", title="Top 10 Cities by Police Killings")
fig.show()

# ---------- 11. Deaths over time ----------
df_fatalities["date"] = pd.to_datetime(df_fatalities["date"], dayfirst=True, errors="coerce")
df_fatalities["year"] = df_fatalities["date"].dt.year
df_fatalities["month"] = df_fatalities["date"].dt.to_period("M").astype(str)
by_month = df_fatalities.groupby("month").size().reset_index(name="deaths")
fig = px.line(by_month, x="month", y="deaths", title="Police Killings Over Time")
fig.show()

# ---------- 12. Killings per state (choropleth) ----------
by_state = df_fatalities["state"].value_counts().reset_index()
by_state.columns = ["state", "deaths"]
fig = px.choropleth(by_state, locations="state", locationmode="USA-states",
                    color="deaths", scope="usa", title="Police Killings by State")
fig.show()

# NOTE: raw counts are misleading. Next step: divide by state population
# to get a per-capita rate before drawing conclusions.