from pathlib import Path

import pandas as pd

YEAR_FILTER = [
    2024,
    2023,
    2022,
    2021,
    2020,
    2019,
    2018,
    2017,
    2016,
    2015,
    2014,
    2013,
    2012,
    2011,
    2010,
]
FILTERS = {
    "london": ["ltds"],
    "uk": ["nts"],
    "dutch": ["odin"],
    "us": ["nhts"],
    "victoria": ["vista"],
    "queensland": ["qhts"],
    "chicago": ["cmap"],
    "kr": ["ktdb"],
}
in_root = Path("~/Projects/foundata/output").expanduser()
write_root = Path("tmp/foundation")
out = write_root / "world"
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

all_attributes.to_csv(out / "training_attributes.csv", index=False)
all_activities.to_csv(out / "training_activities.csv", index=False)

for source_name, sources in FILTERS.items():
    attributes = all_attributes[all_attributes["source"].isin(sources)]
    pids = attributes["pid"].unique()
    activities = all_activities[all_activities["pid"].isin(pids)]

    # get latest 2 years of data for the source
    years = attributes["year"].unique()
    selected_years = sorted(years)[-2:]

    attributes = attributes[attributes["year"].isin(selected_years)]
    pids = attributes["pid"].unique()
    activities = activities[activities["pid"].isin(pids)]

    years = attributes["year"].unique()
    years = "_".join([str(y) for y in years])

    print(f"\n-> Filtered for {source_name} {years}:")

    attributes.to_csv(out / f"{source_name}_attributes.csv", index=False)
    activities.to_csv(out / f"{source_name}_activities.csv", index=False)

    n_atts = attributes.shape[0]
    n_acts = activities.shape[0]

    print(f"Number of attributes in {years} {source_name}: {n_atts}")
    print(f"Number of activities in {years} {source_name}: {n_acts}")
