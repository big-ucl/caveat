from pathlib import Path

import pandas as pd

root = Path("tmp/foundation/home_based")
out = root / "filter"
out.mkdir(parents=True, exist_ok=True)

all_attributes = pd.read_csv(root / "binned_attributes.csv")
all_activities = pd.read_csv(root / "activities.csv")

n_people = all_attributes.shape[0]
n_activities = all_activities.shape[0]

print(f"Total number of people: {n_people}")
print(f"Total number of activities: {n_activities}")

# filter for nts only
nts_attributes = all_attributes[all_attributes["source"] == "nts"]
pids = nts_attributes["pid"].unique()
nts_activities = all_activities[all_activities["pid"].isin(pids)]

nts_attributes.to_csv(out / "nts_attributes.csv", index=False)
nts_activities.to_csv(out / "nts_activities.csv", index=False)

n_atts_nts = nts_attributes.shape[0]
n_activities_nts = nts_activities.shape[0]

print(f"Number of NTS attributes: {n_atts_nts}")
print(f"Number of NTS activities: {n_activities_nts}")

# filter for 2023 only
attributes_2023 = all_attributes[all_attributes["year"] == 2023]
pids_2023 = attributes_2023["pid"].unique()
activities_2023 = all_activities[all_activities["pid"].isin(pids_2023)]

attributes_2023.to_csv(out / "attributes_2023.csv", index=False)
activities_2023.to_csv(out / "activities_2023.csv", index=False)

n_atts_2023 = attributes_2023.shape[0]
n_activities_2023 = activities_2023.shape[0]

print(f"Number of attributes in 2023: {n_atts_2023}")
print(f"Number of activities in 2023: {n_activities_2023}")

# filter for nts 2023 only
nts_attributes_2023 = nts_attributes[nts_attributes["year"] == 2023]
pids_2023 = nts_attributes_2023["pid"].unique()
nts_activities_2023 = nts_activities[nts_activities["pid"].isin(pids_2023)]

nts_attributes_2023.to_csv(out / "nts_attributes_2023.csv", index=False)
nts_activities_2023.to_csv(out / "nts_activities_2023.csv", index=False)

n_atts_2023 = nts_attributes_2023.shape[0]
n_activities_2023 = nts_activities_2023.shape[0]

print(f"Number of NTS attributes in 2023: {n_atts_2023}")
print(f"Number of NTS activities in 2023: {n_activities_2023}")
