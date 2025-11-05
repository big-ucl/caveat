import pickle

import pandas as pd
import torch
from pam.activity import Activity, Leg, Plan
from pam.core import Household, Person
from pam.utils import minutes_to_datetime
from pandas import DataFrame
from pytorch_lightning import Trainer

from caveat import models
from caveat.data import build_latent_conditional_dataloader


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

    def __call__(self, synthetics):
        return self.gen(synthetics)

    def gen(self, synthetics):
        return trim(stretch(pad(self._gen(synthetics))))

    def _gen(self, synthetics):
        synthetic_conditionals = self.attributes_encoder.encode(synthetics)

        dataloader = build_latent_conditional_dataloader(
            synthetic_conditionals,
            self.latent_size,
            len(synthetic_conditionals),
        )

        device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

        batch = dataloader.dataset[0]
        # move data to device
        batch = (batch[0].to(device), batch[1].to(device))

        labels, predictions, zs = self.model.predict_step(batch)

        schedules = self.schedule_encoder.decode(predictions.cpu())
        return schedules


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


class ImposterGame:
    def __init__(
        self,
        generator,
        real_schedules: pd.DataFrame,
        real_labels: pd.DataFrame,
        target_labels: list = [
            "pid",
            "gender",
            "age_group",
            "car_access",
            "work_status",
            "income",
            "year",
        ],
        sample_size: int = 512,
        round_size: int = 4,
    ):
        self.real_schedules = real_schedules
        target_labels = real_labels[target_labels].sample(sample_size)
        self.synthetics_schedules = generator.gen(target_labels)
        self.round_size = round_size
        self.history = []

    def play(self):
        self.offset = pd.Series(range(self.round_size)).sample(1).values[0]

        # sample real schedules
        pids = pd.Series(self.real_schedules.pid.unique()).sample(
            self.round_size - 1
        )
        real_schedules = self.real_schedules[self.real_schedules.pid.isin(pids)]

        pids_first = pd.Series(real_schedules.pid.unique()).sample(self.offset)
        first = real_schedules[real_schedules.pid.isin(pids_first)]
        last = real_schedules[~real_schedules.pid.isin(pids_first)]

        synthetic_pids = pd.Series(
            self.synthetics_schedules.pid.unique()
        ).sample(1)
        synthetics = self.synthetics_schedules[
            self.synthetics_schedules.pid.isin(synthetic_pids)
        ]
        synthetics.loc[:, "pid"] = -1

        population = pd.concat([first, synthetics, last])
        pids = population.pid.unique()
        mapper = {pid: i + 1 for i, pid in enumerate(pids)}
        population.pid = population.pid.map(mapper)

        return Question(schedules=population, answer=self.offset, parent=self)

    def score(self):
        if len(self.history) == 0:
            return None
        return sum(self.history) / len(self.history)

    def report(self):
        score = self.score()
        if score is None:
            print("No rounds played yet.")
        else:
            print(f"Score: {score*100:.2f}% over {len(self.history)} rounds.")

    def reset_history(self):
        self.history = []


class Question:
    def __init__(
        self, schedules: pd.DataFrame, answer: int, parent: ImposterGame
    ):
        self.answer = answer
        self.parent = parent
        plot(schedules)

    def guess(self, location: int):
        if location is None or (
            location < 1 or location > self.parent.round_size
        ):
            print(f"Location must be between 1 and {self.parent.round_size}")
            return
        elif location == self.answer + 1:
            self.parent.history.append(True)
            print("Correct!")
            print(self.parent.report())
        else:
            self.parent.history.append(False)
            print(f"{location} is wrong! Correct answer is: {self.answer + 1}")
            print(self.parent.report())
