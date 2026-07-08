import warnings
from typing import Iterable, Optional

import numpy as np
import polars as pl
from robin.encoders.column_encoders.base import BaseEncoder
from robin.encoders.utils import inverse_token_frequency
from sklearn.mixture import BayesianGaussianMixture
from sklearn.preprocessing import PowerTransformer, StandardScaler
from torch import Tensor


class NonTransform:
    """Identity transform: z = x; inverse: x = z"""

    def fit(self, x):
        return self

    def encode(self, x):
        return x

    def decode(self, x):
        return x

    def log_abs_det_jacobian(self, x):
        return 0


class MinMaxTransform:
    """z = (x - min) / (max - min); inverse: x = z * (max - min) + min"""

    def fit(self, x):
        self.min = np.min(x, axis=0)
        self.max = np.max(x, axis=0)
        return self

    def encode(self, x):
        return (x - self.min) / (self.max - self.min)

    def decode(self, z):
        return z * (self.max - self.min) + self.min

    def log_abs_det_jacobian(self, x):
        # Linear transform z = (x - min)/(max - min) => |det J| = prod(1/(max-min))
        # log|det J| is constant per row
        s = self.max - self.min
        return np.full(x.shape[0], -np.sum(np.log(np.abs(s))))


class StandardScalerTransformer:
    def __init__(self):
        self.ss = StandardScaler(with_mean=True, with_std=True)

    def fit(self, x):
        self.ss.fit(x)
        return self

    def encode(self, x):
        return self.ss.transform(x)

    def decode(self, z):
        return self.ss.inverse_transform(z)

    def log_abs_det_jacobian(self, x):
        # Linear transform z = (x-mu)/sigma => |det J| = prod(1/sigma_j)
        # log|det J| is constant per row
        s = self.ss.scale_
        return np.full(x.shape[0], -np.sum(np.log(np.abs(s))))


class LogTransform:
    """z = log(x + c); inverse: x = exp(z) - c"""

    def fit(self, x):
        # c must be > -min(x) so that x + c > 0 for all x.
        self.c = -np.min(x) + 1e-8
        return self

    def encode(self, x):
        return np.log(x + self.c)

    def decode(self, z):
        return np.exp(z) - self.c

    def log_abs_det_jacobian(self, x):
        return -np.log(x + self.c).sum(axis=1)


class BoxCoxTransform:
    """Uses sklearn PowerTransformer (box-cox)."""

    def __init__(self):
        self.pt = PowerTransformer(method="box-cox", standardize=False)

    def fit(self, x):
        self.pt.fit(x)
        return self

    def encode(self, x):
        return self.pt.transform(x)

    def decode(self, z):
        return self.pt.inverse_transform(z)

    def log_abs_det_jacobian(self, x):
        out = np.zeros(x.shape[0])
        for j, lam in enumerate(self.pt.lambdas_):
            out += (lam - 1.0) * np.log(x[:, j])
        return out


class YeoJohnsonTransform:
    """Uses sklearn PowerTransformer (yeo-johnson)."""

    def __init__(self):
        self.pt = PowerTransformer(method="yeo-johnson", standardize=False)

    def fit(self, x):
        self.pt.fit(x)
        return self

    def encode(self, x):
        return self.pt.transform(x)

    def decode(self, z):
        return self.pt.inverse_transform(z)

    def log_abs_det_jacobian(self, x):
        out = np.zeros(x.shape[0])
        for j, lam in enumerate(self.pt.lambdas_):
            col = x[:, j]
            pos = col >= 0

            # positive
            out[pos] += (lam - 1) * np.log(col[pos] + 1)
            # negative
            out[~pos] += (1 - lam) * np.log(1 - col[~pos])

        return out


class MetaDecomposer(BaseEncoder):
    transformers = [
        NonTransform,
        MinMaxTransform,
        StandardScalerTransformer,
        LogTransform,
        # BoxCoxTransform,
        # YeoJohnsonTransform,
    ]

    def __init__(
        self,
        data: pl.Series,
        name: str,
        max_components: int = 10,
        verbose: bool = False,
        learn_rounding=False,
        enforce_min_max=False,
        use_token_weights: bool = False,
        max_iter: int = 100,
        weight_threshold=0.005,
        seed: Optional[int] = None,
        **kwargs,
    ):
        self.name = name
        self.verbose = verbose
        self._learn_rounding = learn_rounding
        self.enforce_min_max = enforce_min_max
        self.use_token_weights = use_token_weights
        self.max_iter = max_iter
        self.max_components = max_components
        self.weight_threshold = weight_threshold
        self.seed = seed if seed is not None else 12345
        self.size = None
        self.encoding = "decomposed"
        self.slot_size = 2

    def fit_and_encode(self, data: pl.Series) -> Tensor:
        gmm_kwargs = dict(
            name=self.name,
            max_components=self.max_components,
            verbose=self.verbose,
            learn_rounding=self._learn_rounding,
            enforce_min_max=self.enforce_min_max,
            max_iter=self.max_iter,
            weight_threshold=self.weight_threshold,
            seed=self.seed,
        )
        candidates = []
        for transformer_cls in self.transformers:
            transformer = transformer_cls()
            try:
                transformer.fit(data.to_numpy().reshape(-1, 1))
                transformed_data = transformer.encode(
                    data.to_numpy().reshape(-1, 1)
                )
                transformed_series = pl.Series(transformed_data.flatten())
                gmm_encoder = GMMEncoder(**gmm_kwargs)
                gmm_encoder.fit_and_encode(transformed_series)
                score = gmm_encoder.score(transformed_series)
                candidates.append((transformer, gmm_encoder, score))
                if self.verbose:
                    print(
                        f">>> MetaDecomposer: {transformer_cls.__name__} scored {score:.4f} for {self.name} <<<"
                    )
            except Exception as e:
                if self.verbose:
                    print(
                        f">>> MetaDecomposer: {transformer_cls.__name__} failed for {self.name}: {e} <<<"
                    )

        if candidates:
            self.transformer, gmm_encoder, best_score = max(
                candidates, key=lambda c: c[2]
            )
            encoded, weights = gmm_encoder.encode(
                pl.Series(
                    self.transformer.encode(
                        data.to_numpy().reshape(-1, 1)
                    ).flatten()
                )
            )
            if self.verbose:
                print(
                    f">>> MetaDecomposer: Selected {self.transformer.__class__.__name__} "
                    f"(score={best_score:.4f}) for {self.name} <<<"
                )
        else:
            if self.verbose:
                print(
                    f">>> MetaDecomposer: All transforms failed for {self.name}, using raw data <<<"
                )
            self.transformer = NonTransform()
            gmm_encoder = GMMEncoder(**gmm_kwargs)
            encoded, weights = gmm_encoder.fit_and_encode(data)

        self.gmm_encoder = gmm_encoder
        self.threshold_mask = gmm_encoder.threshold_mask
        self._token_weights = gmm_encoder._token_weights

        if self.use_token_weights and self.verbose:
            print(
                f">>> GMMEncoder: Using token weights for {self.name} with shape {self._token_weights.shape} <<<"
            )

        self.dtype = data.dtype
        self.size = sum(self.threshold_mask)

        return encoded, weights

    def encode(self, data: pl.Series) -> Tensor:
        transformed = self.transformer.encode(data.to_numpy().reshape(-1, 1))
        return self.gmm_encoder.encode(pl.Series(transformed.flatten()))

    def decode(self, data) -> pl.Series:
        decoded_transformed = self.gmm_encoder.decode(data)
        raw = self.transformer.decode(
            decoded_transformed.to_numpy().reshape(-1, 1)
        )
        return pl.Series(raw.flatten())


class GMMEncoder(BaseEncoder):
    """Encoder (Transformer) for numerical data using a Bayesian Gaussian Mixture Model.

    This transformation takes a numerical value and transforms it using a Bayesian GMM
    model. It generates two outputs, a discrete value which indicates the selected
    'component' of the GMM and a continuous value which represents the normalized value
    based on the mean and std of the selected component.

    Args:
        learn_rounding_scheme (bool):
            Whether or not to learn what place to round to based on the data seen during ``fit``.
            If ``True``, the data returned by ``reverse_transform`` will be rounded to that place.
            Defaults to ``False``.
        enforce_min_max_values (bool):
            Whether or not to clip the data returned by ``reverse_transform`` to the min and
            max values seen during ``fit``. Defaults to ``False``.
        max_clusters (int):
            The maximum number of mixture components. Depending on the data, the model may select
            fewer components (based on the ``weight_threshold``).
            Defaults to 10.
        weight_threshold (int, float):
            The minimum value a component weight can take to be considered a valid component.
            ``weights_`` under this value will be ignored.
            Defaults to 0.005.

    Attributes:
        transformer:
            An instance of sklearn`s ``BayesianGaussianMixture`` class.
        threshold_indices:
            An array indicating the valid components. If the weight of a component is greater
            than the ``weight_threshold``, it's indicated with True, otherwise it's set to False.
    """

    def __init__(
        self,
        name: str,
        max_components: int = 10,
        verbose: bool = False,
        learn_rounding=False,
        enforce_min_max=False,
        max_iter: int = 100,
        weight_threshold=0.005,
        seed: Optional[int] = None,
        **kwargs,
    ):
        self.name = name
        self.verbose = verbose
        self._learn_rounding = learn_rounding
        self.enforce_min_max = enforce_min_max
        self.max_iter = max_iter
        self.max_components = max_components
        self.weight_threshold = weight_threshold
        self.seed = seed if seed is not None else 12345
        self.size = None

        self.encoding = "decomposed"
        self.slot_size = 2

    def fit_and_encode(self, data: pl.Series) -> Tensor:
        if not data.dtype.is_numeric():
            raise ValueError("GMM Decomposer only supports numeric data types.")
        self._fit(data)
        encoded, weights = self.encode(data)

        self.dtype = data.dtype
        self.size = sum(self.threshold_mask)

        return encoded, weights

    def __str__(self):
        return f"{self.__class__.__name__}: ({self.name}) {self.size}/{self.max_components} components."

    def _fit(self, data: pl.Series):
        """Fit the transformer to the data.

        Args:
            data (pandas.Series):
                Data to fit to.
        """
        self.transformer = BayesianGaussianMixture(
            n_components=self.max_components,
            max_iter=self.max_iter,
            weight_concentration_prior_type="dirichlet_process",
            weight_concentration_prior=None,
            random_state=self.seed,
        )

        if self.enforce_min_max:
            self._min_value = data.min()
            self._max_value = data.max()

        if self._learn_rounding:
            self._rounding_digits = self.get_rounding(data)

        with warnings.catch_warnings():
            warnings.simplefilter("ignore")
            self.transformer.fit(data.to_numpy().reshape(-1, 1))

        self.threshold_mask = self.transformer.weights_ > self.weight_threshold

    def score(self, data: pl.Series) -> float:
        """Calculate the Log Likelihood for the fitted transformation.
        Args:
            data (pl.Series):
                Data to calculate the log likelihood for.

        Returns:
            float: The Log Likelihood value.
        """
        data = data.to_numpy().reshape(-1, 1)
        return self.transformer.score(X=data)

    def encode(self, data: pl.Series) -> Tensor:
        """Transform the numerical data.

        Args:
            data (pl.Series):
                Data to transform.

        Returns:
            torch.Tensor.
        """
        data = data.to_numpy().reshape(-1, 1)

        means = self.transformer.means_.reshape((1, self.max_components))
        means = means[:, self.threshold_mask]
        vars = self.transformer.covariances_.reshape((1, self.max_components))
        stds = np.sqrt(vars)
        stds = stds[:, self.threshold_mask]
        # Multiply stds by 4 so that a value will be in the range [-1,1] with 99.99% probability
        normalized_values = (data - means) / (4 * stds)
        component_probs = self.transformer.predict_proba(data)
        component_probs = component_probs[:, self.threshold_mask]
        # component_probs = (component_probs) / component_probs.sum(
        #     axis=1, keepdims=True
        # )

        r = np.expand_dims(np.random.rand(len(data)), axis=1)
        selected_component = (component_probs.cumsum(axis=1) > r).argmax(axis=1)

        aranged = np.arange(len(data))
        normalized = normalized_values[aranged, selected_component]
        normalized = normalized.reshape([-1, 1])
        normalized = np.clip(normalized, -0.99, 0.99)
        normalized = normalized[:, 0]
        rows = [normalized, selected_component]
        encoded = np.stack(rows, axis=1)
        encoded = Tensor(encoded)
        assert encoded.shape[1] == 2

        if self._token_weights is None:
            self._token_weights = inverse_token_frequency(encoded[:, 1].long())
            # if token weights size is less than max components, pad with zeros?
            if self.verbose:
                print(
                    f">>> GMMEncoder: Using token weights for {self.name} with shape {self._token_weights} <<<"
                )
        weights = self._token_weights.unsqueeze(-1)[encoded[:, 1].long()]
        return encoded, weights

    def decode(self, data: Iterable) -> pl.Series:
        """Convert data back into the original format.

        Args:
            data (Iterable): Data to transform.

        Returns:
            pl.Series.
        """
        assert data.shape[1] == 2
        data = np.asarray(data)

        means = self.transformer.means_.reshape([-1])
        stds = np.sqrt(self.transformer.covariances_).reshape([-1])

        # first col [:,0] is component value, second col [:,1] is component index
        selected_component = data[:, 1]
        selected_component = selected_component.round().astype(int)
        selected_component = selected_component.clip(
            0, self.threshold_mask.sum() - 1
        )

        normalized = np.clip(data[:, 0], -1, 1)

        std_t = stds[self.threshold_mask]
        std_t = std_t[selected_component]
        mean_t = means[self.threshold_mask][selected_component]
        decoded = normalized * 4 * std_t + mean_t

        if self.enforce_min_max:
            decoded = decoded.clip(self._min_value, self._max_value)

        if self._learn_rounding and self._rounding_digits is not None:
            decoded = decoded.round(self._rounding_digits)

        return pl.Series(decoded)
