from pathlib import Path

import pandas as pd

YEAR_FILTERS = {
    "24": [2024],
    "23-24": [2024, 2023],
    "20-24": [2024, 2023, 2022, 2021, 2020],
    "10-24": [
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
    ],
    "00-24": [
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
        2009,
        2008,
        2007,
        2006,
        2005,
        2004,
        2003,
        2002,
        2001,
        2000,
    ],
}
FILTERS = {
    "london": ["ltds"],
    "uk": ["ltds", "nts"],
    "europe": ["ltds", "nts", "odin"],
    "west": ["ltds", "nts", "odin", "nhts", "cmap"],
    "global": ["ltds", "nts", "odin", "nhts", "cmap", "ktdb"],
}
in_root = Path("~/Projects/foundata/output/world").expanduser()
write_root = Path("tmp/foundation")
out = write_root / "experiment"
out.mkdir(parents=True, exist_ok=True)

all_attributes = pd.read_csv(
    in_root / "binned_attributes.csv", low_memory=False
)
all_activities = pd.read_csv(in_root / "activities.csv", low_memory=False)

n_people = all_attributes.shape[0]
n_activities = all_activities.shape[0]

print(f"Total number of people: {n_people}")
print(f"Total number of activities: {n_activities}")

all_attributes.to_csv(out / "all_attributes.csv", index=False)
all_activities.to_csv(out / "all_activities.csv", index=False)

for years_name, years in YEAR_FILTERS.items():
    _attributes = all_attributes[all_attributes["year"].isin(years)]
    pids = _attributes["pid"].unique()
    _activities = all_activities[all_activities["pid"].isin(pids)]

    for source_name, sources in FILTERS.items():
        print(f"\n-> Filtered for {years_name} {source_name}:")

        attributes = _attributes[_attributes["source"].isin(sources)]
        pids = attributes["pid"].unique()
        activities = _activities[_activities["pid"].isin(pids)]

        attributes.to_csv(
            out / f"{years_name}_{source_name}_attributes.csv", index=False
        )
        activities.to_csv(
            out / f"{years_name}_{source_name}_activities.csv", index=False
        )

        n_atts = attributes.shape[0]
        n_acts = activities.shape[0]

        print(f"Number of attributes in {years_name} {source_name}: {n_atts}")
        print(f"Number of activities in {years_name} {source_name}: {n_acts}")
