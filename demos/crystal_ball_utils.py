import pickle

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import torch
from pam.activity import Activity, Leg, Plan
from pam.core import Household, Person
from pam.utils import minutes_to_datetime
from pandas import DataFrame
from pytorch_lightning import Trainer
from scipy.stats import norm

from caveat import models


def to_datetime(minutes: int):
    return minutes_to_datetime(minutes)


def build_plan(schedule: pd.DataFrame):
    plan = Plan()
    try:
        for _, row in schedule.iterrows():
            start, end = to_datetime(row.start), to_datetime(row.end)
            plan.add(Activity(act=row.act, start_time=start, end_time=end))
            plan.add(Leg(mode="", start_time=end, end_time=end, distance=0))
        plan.day.pop(-1)
        return plan
    except Exception as e:
        print(e)
        return None


def plot(schedules: pd.DataFrame):
    hh = Household(0)
    for pid, schedule in schedules.groupby(schedules.pid):
        plan = build_plan(schedule)
        if plan is None:
            continue
        person = Person(pid)
        person.plan = plan
        hh.add(person)
    hh.plot()


class Generator:
    def __init__(
        self,
        model_name: str,
        ckpt_path: str,
        schedule_encoder_path: str,
        attributes_encoder_path: str,
        latent_size: int = 6,
    ) -> None:
        # load model from checkpoint
        self.model = models.library[model_name].load_from_checkpoint(ckpt_path)
        self.latent_size = latent_size
        self.trainer = Trainer()

        # load encoders
        with open(schedule_encoder_path, "rb") as f:
            self.schedule_encoder = pickle.load(f)

        with open(attributes_encoder_path, "rb") as f:
            self.attributes_encoder = pickle.load(f)

        self.ckpt_path = ckpt_path
        self.z = None
        self.device = torch.device(
            "cuda" if torch.cuda.is_available() else "cpu"
        )

    def embed(self, schedule, labels):
        # encode data
        encoded_labels, _ = self.attributes_encoder.encode(labels)

        encoded_schedule = self.schedule_encoder.encode(
            schedules=schedule, labels=encoded_labels, label_weights=None
        )

        (x, _), _, (y, _) = encoded_schedule[0]
        x = x.unsqueeze(0)
        y = y.unsqueeze(0)
        # move data to device
        mu, logvar = self.model.encode(
            x.to(self.device), labels=y.to(self.device)
        )
        zs = self.model.reparameterize(mu, logvar)
        self.z = zs
        self.plot_embedding()

    def plot_embedding(self):
        z = self.z.squeeze().cpu().detach().numpy()
        fig, ax = plt.subplots(figsize=(5, 5))
        for i in range(self.latent_size):
            ax.hist(z[i], bins=30, density=False, alpha=0.6, color="g")
        # plot normal distribution
        mu, std = 0, 1
        xmin, xmax = -3, 3
        x = np.linspace(xmin, xmax, 100)
        p = norm.pdf(x, mu, std)
        ax.plot(x, p, "k", linewidth=2)
        ax.set_title("Latent Samples")
        plt.tight_layout()
        plt.show()

    def __call__(self, synthetics):
        return self.gen(synthetics)

    def gen(self, synthetics):
        return trim(stretch(pad(self._gen(synthetics))))

    def _gen(self, labels):
        labels = pd.concat([labels, labels])
        synthetic_labels, _ = self.attributes_encoder.encode(labels)
        predictions = self.model.decode(
            self.z, labels=synthetic_labels.to(self.device)
        )
        schedules = self.schedule_encoder.decode(predictions.cpu())
        return schedules[schedules.pid == 0]


def stretch(schedules):
    return schedules.groupby(schedules.pid).apply(stretcher)


def stretcher(schedule):
    duration = schedule.duration.sum()
    if duration != 1440:
        a = 1440 / duration
        schedule.duration = (schedule.duration * a).astype(int)
        accumulated = list(schedule.duration.cumsum())
        schedule.start = [0] + accumulated[:-1]
        schedule.end = accumulated
    return schedule


def trim(schedules):
    schedules[schedules.end > 1440] = 1440
    schedules[schedules.start > 1440] = 1440
    schedules.duration = schedules.end - schedules.start
    schedules = schedules[schedules.duration > 0]
    return schedules


def pad(schedules):
    return (
        schedules.groupby(schedules["pid"]).apply(padder).reset_index(drop=True)
    )


def padder(schedule):
    if schedule.end.iloc[-1] < 1440 and schedule.act.iloc[-1] != "home":
        pid = schedule.pid.iloc[0]
        schedule = pd.concat(
            [
                schedule,
                DataFrame(
                    {
                        "pid": pid,
                        "start": schedule.end.iloc[-1],
                        "end": 1440,
                        "duration": 1440 - schedule.end.iloc[-1],
                        "act": "home",
                    },
                    index=[0],
                ),
            ]
        )
    elif schedule.end.iloc[-1] < 1440:
        schedule.end.iloc[-1] = 1440
        schedule.duration.iloc[-1] = 1440 - schedule.start.iloc[-1]
    return schedule
