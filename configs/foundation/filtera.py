from pathlib import Path

import pandas as pd

SOURCE = "nts"

root = Path("tmp/foundation")
out = root / "experiment_a"
out.mkdir(parents=True, exist_ok=True)

all_attributes = pd.read_csv(root / "binned_attributes.csv", low_memory=False)
all_activities = pd.read_csv(root / "activities.csv", low_memory=False)

n_people = all_attributes.shape[0]
n_activities = all_activities.shape[0]

print(f"Total number of people: {n_people}")
print(f"Total number of activities: {n_activities}")

# filter for nts only
attributes_source = all_attributes[all_attributes["source"] == SOURCE]
pids_source = attributes_source["pid"].unique()
activities_source = all_activities[all_activities["pid"].isin(pids_source)]

attributes_source.to_csv(
    out / f"{SOURCE}_attributes_all_years.csv", index=False
)
activities_source.to_csv(
    out / f"{SOURCE}_activities_all_years.csv", index=False
)

n_atts_source = attributes_source.shape[0]
n_activities_source = activities_source.shape[0]

print(f"Number of {SOURCE.upper()} attributes: {n_atts_source}")
print(f"Number of {SOURCE.upper()} activities: {n_activities_source}")

for years in [
    [2024],
    [2024, 2023],
    [2024, 2023, 2022, 2021],
    [2024, 2023, 2022, 2021, 2020, 2019, 2018, 2017],
    [2024, 2023, 2022, 2021, 2020, 2019, 2018, 2017]
    + [2016, 2015, 2014, 2013, 2012, 2011, 2010, 2009],
]:
    print(f"\n-> Filter for {years}")
    attributes_year = attributes_source[attributes_source["year"].isin(years)]
    pids_source_year = attributes_year["pid"].unique()
    activities_year = activities_source[
        activities_source["pid"].isin(pids_source_year)
    ]

    attributes_year.to_csv(
        out / f"{SOURCE}_attributes_{len(years)}years.csv", index=False
    )
    activities_year.to_csv(
        out / f"{SOURCE}_activities_{len(years)}years.csv", index=False
    )

    n_atts_year = attributes_year.shape[0]
    n_activities_year = activities_year.shape[0]

    print(f"Number of attributes in {years}: {n_atts_year}")
    print(f"Number of activities in {years}: {n_activities_year}")
