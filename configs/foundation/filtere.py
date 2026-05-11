from pathlib import Path

import pandas as pd

YEAR = 2017
SOURCE = "vista"
REQUIRED = ["sex", "hh_zone", "avg_speed", "age", "hh_income", "year"]

root = Path("tmp/foundation")
out = root / "experiment_e"
out.mkdir(parents=True, exist_ok=True)

print("Loading")

all_attributes = pd.read_csv(root / "binned_attributes.csv", low_memory=False)
all_activities = pd.read_csv(root / "activities.csv", low_memory=False)

n_people = all_attributes.shape[0]
n_activities = all_activities.shape[0]

print(f"Total number of people: {n_people}")
print(f"Total number of activities: {n_activities}")

print("\n-> Filter for required attributes")
for att in REQUIRED:
    all_attributes = all_attributes[all_attributes[att] != "unknown"]
    pids = all_attributes.pid.unique()
    all_activities = all_activities[all_activities.pid.isin(pids)]

n_people = all_attributes.shape[0]
n_activities = all_activities.shape[0]

print(f"Total number of people: {n_people}")
print(f"Total number of activities: {n_activities}")


# =======================================
print("\n-> Filter for vista target")

attributes_source = all_attributes[
    (all_attributes["source"] == SOURCE) & (all_attributes["year"] >= YEAR)
]
pids_source = attributes_source["pid"].unique()
activities_source = all_activities[all_activities["pid"].isin(pids_source)]


attributes_source.to_csv(out / "target_attributes.csv", index=False)
activities_source.to_csv(out / "target_activities.csv", index=False)

n_atts_source = attributes_source.shape[0]
n_activities_source = activities_source.shape[0]

print(f"Number of target attributes: {n_atts_source}")
print(f"Number of target activities: {n_activities_source}")

# =======================================
print("\n-> Filter for qhts")

SOURCE = "qhts"

attributes_source = all_attributes[
    (all_attributes["source"] == SOURCE) & (all_attributes["year"] >= YEAR)
]
pids_source = attributes_source["pid"].unique()
activities_source = all_activities[all_activities["pid"].isin(pids_source)]


attributes_source.to_csv(out / "qhts_attributes.csv", index=False)
activities_source.to_csv(out / "qhts_activities.csv", index=False)

n_atts_source = attributes_source.shape[0]
n_activities_source = activities_source.shape[0]

print(f"Number of target attributes: {n_atts_source}")
print(f"Number of target activities: {n_activities_source}")

# =======================================
print("\n-> Filter for western")

SOURCES = ["qhts", "nts", "ltds", "nhts", "cmap"]

attributes_source = all_attributes[
    (all_attributes["source"].isin(SOURCES)) & (all_attributes["year"] >= YEAR)
]
pids_source = attributes_source["pid"].unique()
activities_source = all_activities[all_activities["pid"].isin(pids_source)]


attributes_source.to_csv(out / "western_attributes.csv", index=False)
activities_source.to_csv(out / "western_activities.csv", index=False)

n_atts_source = attributes_source.shape[0]
n_activities_source = activities_source.shape[0]

print(f"Number of target attributes: {n_atts_source}")
print(f"Number of target activities: {n_activities_source}")

# =======================================
print("\n-> Filter for western no aus")

SOURCES = ["nts", "ltds", "nhts", "cmap"]

attributes_source = all_attributes[
    (all_attributes["source"].isin(SOURCES)) & (all_attributes["year"] >= YEAR)
]
pids_source = attributes_source["pid"].unique()
activities_source = all_activities[all_activities["pid"].isin(pids_source)]


attributes_source.to_csv(out / "western2_attributes.csv", index=False)
activities_source.to_csv(out / "western2_activities.csv", index=False)

n_atts_source = attributes_source.shape[0]
n_activities_source = activities_source.shape[0]

print(f"Number of target attributes: {n_atts_source}")
print(f"Number of target activities: {n_activities_source}")
