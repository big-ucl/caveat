import pandas as pd


class TargetLabelSampler:
    def __init__(self, target_labels: pd.DataFrame, target_columns: list):
        assert "pid" in target_labels.columns
        assert all(column in target_labels.columns for column in target_columns)

        self.target_columns = target_columns
        # self.label_dtypes = target_labels.dtypes.to_dict()
        # self.target_labels = target_labels[target_columns].copy()

        self.target_labels_dict = {
            k: v for k, v in target_labels.groupby(target_columns)
        }
        self.target_sizes = {
            k: len(v) for k, v in self.target_labels_dict.items()
        }
        self.found_sizes = {k: 0 for k in self.target_labels_dict.keys()}

        self.n = len(target_labels)
        self.sampled_schedules = []
        self.sampled_labels = []

        self.sample_n = 0
        self.i = 0

    def nfound(self):
        return sum(self.found_sizes.values())

    def sample(self, labels, schedules):
        assert "pid" in labels.columns
        assert "pid" in schedules.columns
        assert all(column in labels.columns for column in self.target_columns)

        self.sample_n += 1
        print("Sampling iteration ", self.sample_n)

        labels_dict = {k: v for k, v in labels.groupby(self.target_columns)}

        for target_label, target_data in self.target_labels_dict.items():
            if target_data is None or len(target_data) == 0:
                continue

            found_data = labels_dict.get(target_label)
            if found_data is None or len(found_data) == 0:
                continue

            n_extract = min(len(target_data), len(found_data))
            self.found_sizes[target_label] += n_extract
            found_idx = target_data.index[:n_extract]

            # labels
            sampled_labels = target_data.iloc[:n_extract].copy()
            sampled_pids = sampled_labels["pid"]

            sampled_labels["pid"] = range(self.i, self.i + n_extract)
            self.sampled_labels.append(sampled_labels)

            # schedules
            sampled_schedules = schedules[
                schedules["pid"].isin(sampled_pids)
            ].copy()
            sampled_schedules["pid"] = sampled_schedules.groupby("pid").ngroup()
            sampled_schedules["pid"] += self.i
            self.sampled_schedules.append(sampled_schedules)

            # update
            target_data.drop(found_idx, inplace=True, axis=0)  # is this ok?
            self.i += n_extract

        print()  # not a mistake

    def is_done(self):
        if self.nfound() == self.n:
            return True
        return False

    def print(self, verbose=False):
        perc = self.nfound() / self.n
        print(
            f"Sampled {perc:.2%} of target labels in {self.sample_n} sampling iterations.)"
        )
        if verbose:
            print("Unsampled_labels:")
            for i, n in self.target_sizes.items():
                found = self.found_sizes[i]
                if found < n:
                    print(f"<!>{i}: {found / n:.2%} of {n} found.")

    def finish(self):
        if len(self.sampled_labels) == 0:
            return pd.DataFrame(), pd.DataFrame()

        sampled_labels = pd.concat(
            self.sampled_labels, axis=0, ignore_index=True
        )

        sampled_schedules = pd.concat(
            self.sampled_schedules, axis=0, ignore_index=True
        )
        sampled_schedules = sampled_schedules.set_index(
            pd.Series(range(len(sampled_schedules)))
        )
        return (sampled_labels, sampled_schedules)
