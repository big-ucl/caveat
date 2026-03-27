from pathlib import Path

import pandas as pd

YEAR = 2024
YEARS = [2021, 2022, 2023, 2024]
REQUIRED = []

root = Path("tmp/foundation")
out = root / "experiment_b"
out.mkdir(parents=True, exist_ok=True)

all_attributes = pd.read_csv(root / "binned_attributes.csv", low_memory=False)
all_activities = pd.read_csv(root / "activities.csv", low_memory=False)

n_people = all_attributes.shape[0]
n_activities = all_activities.shape[0]

print(f"Total number of people: {n_people}")
print(f"Total number of activities: {n_activities}")

print(f"Filter for {REQUIRED} not unknwon")
for required in REQUIRED:
    all_attributes = all_attributes[all_attributes[required] != "unknown"]
pids = all_attributes["pid"].unique()
all_activities = all_activities[all_activities["pid"].isin(pids)]

print(f"Filter for {YEARS}")
all_attributes = all_attributes[all_attributes["year"].isin(YEARS)]
pids = all_attributes["pid"].unique()
all_activities = all_activities[all_activities["pid"].isin(pids)]

n_people = all_attributes.shape[0]
n_activities = all_activities.shape[0]

print(f"Total number of people: {n_people}")
print(f"Total number of activities: {n_activities}")

# baseline; just NTS YEAR

nts_attributes_2024 = all_attributes[
    (all_attributes["year"] == YEAR) & (all_attributes["source"] == "nts")
]
pids = nts_attributes_2024["pid"].unique()
nts_activities_2024 = all_activities[all_activities["pid"].isin(pids)]

n_people = nts_attributes_2024.shape[0]
n_activities = nts_activities_2024.shape[0]

print(f"NTS 2024 number of people: {n_people}")
print(f"NTS 2024 number of activities: {n_activities}")

nts_attributes_2024.to_csv(out / "nts_attributes_2024.csv", index=False)
nts_activities_2024.to_csv(out / "nts_activities_2024.csv", index=False)

for name, selection in {
    "uk": ["ltds"],
    "uk+aus": ["ltds", "vista", "qhts"],
    "west": ["ltds", "nhts", "vista", "qhts", "cmap"],
    "world": ["ltds", "nhts", "vista", "qhts", "cmap", "ktdb"],
}.items():
    print(f"\n-> Filter for {name} (sources: {selection})")
    attributes = all_attributes[all_attributes["source"].isin(selection)]
    pids = attributes["pid"].unique()
    activities = all_activities[all_activities["pid"].isin(pids)]

    # concat nts 2024 to the attributes and activities
    attributes = pd.concat([attributes, nts_attributes_2024], ignore_index=True)
    activities = pd.concat([activities, nts_activities_2024], ignore_index=True)

    attributes.to_csv(out / f"{name}_attributes.csv", index=False)
    activities.to_csv(out / f"{name}_activities.csv", index=False)

    n_atts_year = attributes.shape[0]
    n_activities_year = activities.shape[0]

    print(f"Number of attributes in {name}: {n_atts_year}")
    print(f"Number of activities in {name}: {n_activities_year}")
