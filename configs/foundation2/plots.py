import pandas as pd
from matplotlib import pyplot as plt


def split_on(schedules, attributes, by="work_status"):
    splits = attributes.groupby(by)
    return {
        k: schedules.loc[schedules.pid.isin(list(v.pid))]
        for k, v in splits
        if k != "unknown"
    }


def duration(schedules, act="work"):
    durations = schedules.end - schedules.start
    return durations[schedules.act == act].mean()


def count_trips(schedules):
    n = schedules.pid.nunique()
    return (len(schedules) - n) / n


def count_acts(schedules, act="work"):
    n = schedules.pid.nunique()
    return len(schedules.loc[schedules.act == act]) / n


def _compute_metric(
    schedules_dict, attributes_dict, target_sched, target_atts, group, fn
):
    """Build dataframe for a given metric, grouping, and function."""
    data = {"Target": {}}

    # target first
    for key, split in split_on(target_sched, target_atts, by=group).items():
        data["Target"][key] = fn(split)

    # models
    for name, sched in schedules_dict.items():
        data[name] = {}
        atts = attributes_dict[name]
        for key, split in split_on(sched, atts, by=group).items():
            data[name][key] = fn(split)

    return pd.DataFrame(data)


def plot_a(
    schedules, attributes, target_schedules, target_attributes, figsize=(8, 8)
):
    day_order = [
        "monday",
        "tuesday",
        "wednesday",
        "thursday",
        "friday",
        "saturday",
        "sunday",
    ]
    days = [
        "Monday",
        "Tuesday",
        "Wednesday",
        "Thursday",
        "Friday",
        "Saturday",
        "Sunday",
    ]

    age_order = ["≤18", "18-36", "36-50", "50-64", ">64"]
    ages = ["≤18", "18-36", "36-50", "50-64", ">64"]

    dist_order = [
        "≤3.23556",
        "3.23556-5.84",
        "5.84-9.424",
        "9.424-16.4",
        ">16.4",
    ]
    distances = ["≤3km", "3-5km", "5-9km", "9-16km", ">16km"]

    order_map = {
        "day": (day_order, days),
        "access_egress_distance": (dist_order, distances),
        "age": (age_order, ages),
    }

    colours = [
        "black",
        "cornflowerblue",
        "orange",
        "red",
        "hotpink",
        "green",
        "green",
        "brown",
        "grey",
        "purple",
        "lightgreen",
    ]
    styles = ["-", ":", ":", ":", ":", ":", "-.", ":", ":"]

    # what each subplot represents: (row, col) → (metric_fn, group, ylabel, xlabel, activity)
    plots = [
        # (count_trips, "day", "Trip Frequency", None),
        # (count_trips, "access_egress_distance", None, None),
        # (count_trips, "age", None, None),
        (count_acts, "day", "Working Frequency", None, "work"),
        (count_acts, "access_egress_distance", None, None, "work"),
        (count_acts, "age", None, None, "work"),
        (count_acts, "day", "Shopping Frequency", None, "shop"),
        (count_acts, "access_egress_distance", None, None, "shop"),
        (count_acts, "age", None, None, "shop"),
        (duration, "day", "Working Duration", None, "work"),
        (duration, "access_egress_distance", None, None, "work"),
        (duration, "age", None, None, "work"),
        (duration, "day", "Shopping Duration", "Day", "shop"),
        (duration, "access_egress_distance", None, "PT access/egress", "shop"),
        (duration, "age", None, "Age group", "shop"),
    ]

    fig, axs = plt.subplots(4, 3, figsize=figsize, sharex="col", sharey="row")

    # iterate 15 subplots
    for ax, (fn, group, ylabel, xlabel, act) in zip(axs.flat, plots):
        metric_fn = (lambda s: fn(s, act)) if act else fn
        df = _compute_metric(
            schedules,
            attributes,
            target_schedules,
            target_attributes,
            group,
            metric_fn,
        )

        order, labels = order_map[group]
        df.loc[order].plot(ax=ax, color=colours, style=styles, lw=2)
        ax.legend().remove()

        if ylabel:
            ax.set_ylabel(ylabel)
        if xlabel:
            ax.set_xlabel(xlabel)
            if group == "day":
                ax.set_xticks(range(len(labels)), labels, rotation=90)
            if group == "access_egress_distance":
                ax.set_xticks(range(len(labels)), labels, rotation=90)
            if group == "age":
                ax.set_xticks(range(len(labels)), labels, rotation=90)

    fig.align_ylabels(axs[:, 0])
    fig.align_xlabels(axs[-1, :])
    fig.tight_layout()

    # global legend
    handles, labels = axs.flat[-1].get_legend_handles_labels()
    fig.legend(
        handles,
        labels,
        loc="lower center",
        frameon=False,
        fontsize=10,
        bbox_to_anchor=(1.15, 0.5),
    )

    return fig


def plot_b(
    schedules, attributes, target_schedules, target_attributes, figsize=(8, 8)
):
    day_order = [
        "monday",
        "tuesday",
        "wednesday",
        "thursday",
        "friday",
        "saturday",
        "sunday",
    ]
    days = [
        "Monday",
        "Tuesday",
        "Wednesday",
        "Thursday",
        "Friday",
        "Saturday",
        "Sunday",
    ]

    age_order = ["≤18", "18-36", "36-50", "50-64", ">64"]
    ages = ["≤18", "18-36", "36-50", "50-64", ">64"]

    dist_order = [
        "≤10.0174",
        "10.0174-19.2",
        "19.2-28.8",
        "28.8-42.0438",
        ">42.0438",
    ]
    distances = ["≤10km/hr", "-20km/hr", "-30km/hr", "-40km/hr", ">40km/hr"]

    order_map = {
        "day": (day_order, days),
        "avg_speed": (dist_order, distances),
        "age": (age_order, ages),
    }

    colours = [
        "black",
        "cornflowerblue",
        "green",
        "red",
        "hotpink",
        "orange",
        "lightblue",
        "brown",
        "grey",
        "purple",
        "lightgreen",
    ]
    styles = ["-", ":", ":", ":", ":", ":", "-.", ":", ":"]

    # what each subplot represents: (row, col) → (metric_fn, group, ylabel, xlabel, activity)
    plots = [
        # (count_trips, "day", "Trip Frequency", None),
        # (count_trips, "avg_speed", None, None),
        # (count_trips, "age", None, None),
        (count_acts, "day", "Working Frequency", None, "work"),
        (count_acts, "avg_speed", None, None, "work"),
        (count_acts, "age", None, None, "work"),
        (count_acts, "day", "Shopping Frequency", None, "shop"),
        (count_acts, "avg_speed", None, None, "shop"),
        (count_acts, "age", None, None, "shop"),
        (duration, "day", "Working Duration", None, "work"),
        (duration, "avg_speed", None, None, "work"),
        (duration, "age", None, None, "work"),
        (duration, "day", "Shopping Duration", "Day", "shop"),
        (duration, "avg_speed", None, "Avg. travel speed", "shop"),
        (duration, "age", None, "Age group", "shop"),
    ]

    fig, axs = plt.subplots(4, 3, figsize=figsize, sharex="col", sharey="row")

    # iterate 15 subplots
    for ax, (fn, group, ylabel, xlabel, act) in zip(axs.flat, plots):
        metric_fn = (lambda s: fn(s, act)) if act else fn
        df = _compute_metric(
            schedules,
            attributes,
            target_schedules,
            target_attributes,
            group,
            metric_fn,
        )

        order, labels = order_map[group]
        df.loc[order].plot(ax=ax, color=colours, style=styles, lw=2)
        ax.legend().remove()

        if ylabel:
            ax.set_ylabel(ylabel)
        if xlabel:
            ax.set_xlabel(xlabel)
            if group == "day":
                ax.set_xticks(range(len(labels)), labels, rotation=90)
            if group == "avg_speed":
                ax.set_xticks(range(len(labels)), labels, rotation=90)
            if group == "age":
                ax.set_xticks(range(len(labels)), labels, rotation=90)

    fig.align_ylabels(axs[:, 0])
    fig.align_xlabels(axs[-1, :])
    fig.tight_layout()

    # global legend
    handles, labels = axs.flat[-1].get_legend_handles_labels()
    fig.legend(
        handles,
        labels,
        loc="lower center",
        frameon=False,
        fontsize=10,
        bbox_to_anchor=(1.15, 0.5),
    )

    return fig


def plot_c(
    schedules, attributes, target_schedules, target_attributes, figsize=(8, 3)
):
    income_order = [
        "≤19777",
        "19777-31885",
        "31885-46848",
        "46848-81469",
        ">81469",
    ]
    income = ["Lowest", "Low", "Mid", "High", "Highest"]

    order_map = {"hh_income": (income_order, income)}

    colours = ["black", "cornflowerblue", "red", "green"]
    styles = ["-", "-.", "-.", "-."]

    plots = [
        (count_acts, "hh_income", "Work Frequency", "HH income", "work"),
        (count_acts, "hh_income", "Shop Frequency", "HH income", "shop"),
        (duration, "hh_income", "Work Duration", "HH income", "work"),
        (duration, "hh_income", "Shop Duration", "HH income", "shop"),
    ]

    fig, axs = plt.subplots(1, 4, figsize=figsize, constrained_layout=True)

    # iterate 15 subplots
    for ax, (fn, group, ylabel, xlabel, act) in zip(axs.flat, plots):
        metric_fn = (lambda s: fn(s, act)) if act else fn
        df = _compute_metric(
            schedules,
            attributes,
            target_schedules,
            target_attributes,
            group,
            metric_fn,
        )

        order, labels = order_map[group]
        df.loc[order].plot(ax=ax, color=colours, style=styles, lw=2)
        ax.legend().remove()

        if ylabel:
            ax.set_title(ylabel, fontsize=10)
        if xlabel:
            ax.set_xlabel(xlabel)
            if group == "day":
                ax.set_xticks(range(len(labels)), labels, rotation=90)
            if group == "avg_speed":
                ax.set_xticks(range(len(labels)), labels, rotation=90)
            if group == "hh_income":
                ax.set_xticks(range(len(labels)), labels, rotation=90)

    # global legend
    handles, labels = axs.flat[-1].get_legend_handles_labels()
    fig.legend(
        handles,
        labels,
        loc="lower center",
        frameon=False,
        fontsize=10,
        bbox_to_anchor=(0.5, 1),
    )

    return fig


def plot_e(
    schedules, attributes, target_schedules, target_attributes, figsize=(8, 8)
):
    zone_order = ["urban", "suburban", "rural"]
    zones = ["urban", "suburban", "rural"]

    age_order = ["≤18", "18-36", "36-50", "50-64", ">64"]
    ages = ["≤18", "18-36", "36-50", "50-64", ">64"]

    dist_order = [
        "≤10.0174",
        "10.0174-19.2",
        "19.2-28.8",
        "28.8-42.0438",
        ">42.0438",
    ]
    distances = ["≤10km/hr", "-20km/hr", "-30km/hr", "-40km/hr", ">40km/hr"]

    order_map = {
        "hh_zone": (zone_order, zones),
        "avg_speed": (dist_order, distances),
        "age": (age_order, ages),
    }

    colours = [
        "black",
        "cornflowerblue",
        "green",
        "red",
        "hotpink",
        "orange",
        "lightblue",
        "brown",
        "grey",
        "purple",
        "lightgreen",
    ]
    styles = ["-", ":", ":", ":", ":", ":", "-.", ":", ":"]

    # what each subplot represents: (row, col) → (metric_fn, group, ylabel, xlabel, activity)
    plots = [
        (count_acts, "hh_zone", "Working Frequency", None, "work"),
        (count_acts, "avg_speed", None, None, "work"),
        (count_acts, "age", None, None, "work"),
        (count_acts, "hh_zone", "Shopping Frequency", None, "shop"),
        (count_acts, "avg_speed", None, None, "shop"),
        (count_acts, "age", None, None, "shop"),
        (duration, "hh_zone", "Working Duration", None, "work"),
        (duration, "avg_speed", None, None, "work"),
        (duration, "age", None, None, "work"),
        (duration, "hh_zone", "Shopping Duration", "Zone", "shop"),
        (duration, "avg_speed", None, "Avg. travel speed", "shop"),
        (duration, "age", None, "Age group", "shop"),
    ]

    fig, axs = plt.subplots(4, 3, figsize=figsize, sharex="col", sharey="row")

    # iterate 15 subplots
    for ax, (fn, group, ylabel, xlabel, act) in zip(axs.flat, plots):
        metric_fn = (lambda s: fn(s, act)) if act else fn
        df = _compute_metric(
            schedules,
            attributes,
            target_schedules,
            target_attributes,
            group,
            metric_fn,
        )

        order, labels = order_map[group]
        df.loc[order].plot(ax=ax, color=colours, style=styles, lw=2)
        ax.legend().remove()

        if ylabel:
            ax.set_ylabel(ylabel)
        if xlabel:
            ax.set_xlabel(xlabel)
            if group == "hh_zone":
                ax.set_xticks(range(len(labels)), labels, rotation=90)
            if group == "avg_speed":
                ax.set_xticks(range(len(labels)), labels, rotation=90)
            if group == "age":
                ax.set_xticks(range(len(labels)), labels, rotation=90)

    fig.align_ylabels(axs[:, 0])
    fig.align_xlabels(axs[-1, :])
    fig.tight_layout()

    # global legend
    handles, labels = axs.flat[-1].get_legend_handles_labels()
    fig.legend(
        handles,
        labels,
        loc="lower center",
        frameon=False,
        fontsize=10,
        bbox_to_anchor=(1.15, 0.5),
    )

    return fig


def plot_f(
    schedules, attributes, target_schedules, target_attributes, figsize=(8, 8)
):
    employment_order = ["employed", "student", "retired", "unemployed"]
    employments = ["employed", "student", "retired", "unemployed"]

    age_order = ["≤17", "17-36", "36-51", "51-65", ">65"]
    ages = ["≤17", "17-36", "36-51", "51-65", ">65"]
    income_order = [
        "≤18811",
        "18811-30935",
        "30935-45739",
        "45739-78565",
        ">78565",
    ]
    income = ["Lowest", "Low", "Mid", "High", "Highest"]

    order_map = {
        "employment": (employment_order, employments),
        "hh_income": (income_order, income),
        "age": (age_order, ages),
    }

    colours = [
        "black",
        "cornflowerblue",
        "green",
        "red",
        "hotpink",
        "orange",
        "lightblue",
        "brown",
        "grey",
        "purple",
        "lightgreen",
    ]
    styles = ["-", ":", ":", ":", ":", ":", "-.", ":", ":"]

    # what each subplot represents: (row, col) → (metric_fn, group, ylabel, xlabel, activity)
    plots = [
        (count_acts, "employment", "Working Frequency", None, "work"),
        (count_acts, "hh_income", None, None, "work"),
        (count_acts, "age", None, None, "work"),
        (count_acts, "employment", "Shopping Frequency", None, "shop"),
        (count_acts, "hh_income", None, None, "shop"),
        (count_acts, "age", None, None, "shop"),
        (duration, "employment", "Working Duration", None, "work"),
        (duration, "hh_income", None, None, "work"),
        (duration, "age", None, None, "work"),
        (
            duration,
            "employment",
            "Shopping Duration",
            "Employment Status",
            "shop",
        ),
        (duration, "hh_income", None, "Household Income", "shop"),
        (duration, "age", None, "Age group", "shop"),
    ]

    fig, axs = plt.subplots(4, 3, figsize=figsize, sharex="col", sharey="row")

    # iterate 15 subplots
    for ax, (fn, group, ylabel, xlabel, act) in zip(axs.flat, plots):
        metric_fn = (lambda s: fn(s, act)) if act else fn
        df = _compute_metric(
            schedules,
            attributes,
            target_schedules,
            target_attributes,
            group,
            metric_fn,
        )

        order, labels = order_map[group]
        df.loc[order].plot(ax=ax, color=colours, style=styles, lw=2)
        ax.legend().remove()

        if ylabel:
            ax.set_ylabel(ylabel)
        if xlabel:
            ax.set_xlabel(xlabel)
            if group == "hh_zone":
                ax.set_xticks(range(len(labels)), labels, rotation=90)
            if group == "hh_income":
                ax.set_xticks(range(len(labels)), labels, rotation=90)
            if group == "age":
                ax.set_xticks(range(len(labels)), labels, rotation=90)

    fig.align_ylabels(axs[:, 0])
    fig.align_xlabels(axs[-1, :])
    fig.tight_layout()

    # global legend
    handles, labels = axs.flat[-1].get_legend_handles_labels()
    fig.legend(
        handles,
        labels,
        loc="lower center",
        frameon=False,
        fontsize=10,
        bbox_to_anchor=(1.15, 0.5),
    )

    return fig
