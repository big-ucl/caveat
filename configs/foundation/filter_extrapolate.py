from pathlib import Path

import pandas as pd

YEAR_FILTER = [2024, 2023, 2022, 2021, 2020, 2019]
FILTERS = {
    "base": ["vista"],
    "australia": ["qhts"],
    "western": ["nts", "odin", "nhts", "ltds", "cmap", "qhts", "ktdb"],
}
SOURCE_MAP = {
    "vista": "australia",
    "qhts": "australia",
    "nts": "uk",
    "odin": "europe",
    "nhts": "us",
    "ltds": "uk",
    "cmap": "us",
    "ktdb": "korea",
}
DAY_MAP = {
    "monday": "monday",
    "tuesday": "tuesday",
    "wednesday": "wednesday",
    "thursday": "thursday",
    "friday": "friday",
    "saturday": "weekend",
    "sunday": "weekend",
}

in_root = Path("~/Projects/foundata/output").expanduser()
write_root = Path("tmp/foundation")
out = write_root / "extrapolate"
out.mkdir(parents=True, exist_ok=True)

all_attributes = pd.read_csv(
    in_root / "binned_attributes.csv", low_memory=False
)
all_activities = pd.read_csv(in_root / "activities.csv", low_memory=False)

all_attributes = all_attributes[all_attributes["year"].isin(YEAR_FILTER)]
pids = all_attributes["pid"].unique()
all_activities = all_activities[all_activities["pid"].isin(pids)]

n_people = all_attributes.shape[0]
n_activities = all_activities.shape[0]

print(f"Total number of people: {n_people}")
print(f"Total number of activities: {n_activities}")

for source_name, sources in FILTERS.items():
    attributes = all_attributes[all_attributes["source"].isin(sources)]
    pids = attributes["pid"].unique()
    activities = all_activities[all_activities["pid"].isin(pids)]

    # map sources to their broader source category
    attributes.loc[:, "source"] = attributes.loc[:, "source"].map(SOURCE_MAP)

    attributes.loc[:, "day"] = attributes.loc[:, "day"].map(DAY_MAP)

    print(f"\n-> Filtered for {source_name}:")

    attributes.to_csv(out / f"{source_name}_attributes.csv", index=False)
    activities.to_csv(out / f"{source_name}_activities.csv", index=False)

    n_atts = attributes.shape[0]
    n_acts = activities.shape[0]

    print(f"Number of attributes in {source_name}: {n_atts}")
    print(f"Number of activities in {source_name}: {n_acts}")
