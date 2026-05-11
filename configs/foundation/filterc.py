from pathlib import Path

import pandas as pd

YEAR = 2024
SOURCE = "nts"

root = Path("tmp/foundation")
out = root / "experiment_c"
out.mkdir(parents=True, exist_ok=True)

print("Loading")

all_attributes = pd.read_csv(root / "binned_attributes.csv", low_memory=False)
all_activities = pd.read_csv(root / "activities.csv", low_memory=False)

n_people = all_attributes.shape[0]
n_activities = all_activities.shape[0]

print(f"Total number of people: {n_people}")
print(f"Total number of activities: {n_activities}")

# =======================================
print("\n-> Target is NTS 2024")
attributes_source = all_attributes[
    (all_attributes["source"] == SOURCE) & (all_attributes["year"] == YEAR)
]
pids_source = attributes_source["pid"].unique()
activities_source = all_activities[all_activities["pid"].isin(pids_source)]

attributes_source.to_csv(out / f"{SOURCE}_attributes_{YEAR}.csv", index=False)
activities_source.to_csv(out / f"{SOURCE}_activities_{YEAR}.csv", index=False)

n_atts_source = attributes_source.shape[0]
n_activities_source = activities_source.shape[0]

print(f"Number of {SOURCE.upper()} attributes: {n_atts_source}")
print(f"Number of {SOURCE.upper()} activities: {n_activities_source}")

# =======================================
print(
    "\n-> Temporal transfer (filter for nts only, 8 years, and remove 2024 hh_income data)"
)
YEARS = [2024, 2023, 2022, 2021, 2020, 2019, 2018, 2017]

attributes_source = all_attributes[
    (all_attributes["source"] == SOURCE) & (all_attributes["year"].isin(YEARS))
]
pids_source = attributes_source["pid"].unique()
activities_source = all_activities[all_activities["pid"].isin(pids_source)]

attributes_source.loc[
    (attributes_source["source"] == SOURCE)
    & (attributes_source["year"] == YEAR),
    "hh_income",
] = "unknown"

attributes_source.to_csv(
    out / f"{SOURCE}_attributes_no_{YEAR}_hh_income.csv", index=False
)
activities_source.to_csv(
    out / f"{SOURCE}_activities_no_{YEAR}_hh_income.csv", index=False
)

n_atts_source = attributes_source.shape[0]
n_activities_source = activities_source.shape[0]

print(f"Number of {SOURCE.upper()} attributes: {n_atts_source}")
print(f"Number of {SOURCE.upper()} activities: {n_activities_source}")


# =======================================


YEARS = [2024, 2023, 2022, 2021, 2020, 2019, 2018, 2017]
SOURCES = ["nts", "ltds"]

print(
    f"\n-> Spatial transfer (filter for {SOURCES} schedules last {len(YEARS)} years)"
)

attributes_source = all_attributes[
    (all_attributes["source"].isin(SOURCES))
    & (all_attributes["year"].isin(YEARS))
]
pids_source = attributes_source["pid"].unique()
activities_source = all_activities[all_activities["pid"].isin(pids_source)]

attributes_source.loc[
    attributes_source["source"] == SOURCE, "hh_income"
] = "unknown"

attributes_source.to_csv(
    out / f"uk_attributes_no_{YEAR}_hh_income.csv", index=False
)
activities_source.to_csv(
    out / f"uk_activities_no_{YEAR}_hh_income.csv", index=False
)

n_atts_source = attributes_source.shape[0]
n_activities_source = activities_source.shape[0]

print(f"Number of {SOURCE.upper()} attributes: {n_atts_source}")
print(f"Number of {SOURCE.upper()} activities: {n_activities_source}")
