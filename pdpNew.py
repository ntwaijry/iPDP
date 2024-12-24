import numbers
from itertools import chain

from scipy.stats._mstats_basic import mquantiles
from _p_d import p_d
import numpy as np
from scipy import sparse

from sklearn.inspection import PartialDependenceDisplay
from sklearn.inspection._pd_utils import _check_feature_names, _get_feature_index
from sklearn.base import is_regressor
from sklearn.utils import check_array, _safe_indexing, check_matplotlib_support
from sklearn.utils._encode import _unique
from sklearn.utils.parallel import delayed, Parallel

class pdpNew(PartialDependenceDisplay):

    def __init__(
        self,
        pd_results,
        *,
        features,
        feature_names,
        target_idx,
        deciles,
        pdp_lim="deprecated",
        kind="average",
        subsample=1000,
        random_state=None,
        is_categorical=None,
    ):
        super().__init__(
            pd_results=pd_results,
            features=features,
            feature_names=feature_names,
            target_idx=target_idx,
            deciles=deciles,
            pdp_lim=pdp_lim,
            kind=kind,
            subsample=subsample,
            random_state=random_state,
            is_categorical=is_categorical
        )

    @classmethod
    def from_estimator(
        cls,
        estimator,
        X,
        features,
        *,
        categorical_features=None,
        feature_names=None,
        target=None,
        response_method="auto",
        n_cols=3,
        grid_resolution=100,
        percentiles=(0.05, 0.95),
        method="auto",
        n_jobs=None,
        verbose=0,
        line_kw=None,
        ice_lines_kw=None,
        pd_line_kw=None,
        contour_kw=None,
        ax=None,
        kind="average",
        centered=False,
        subsample=1000,
        random_state=None,
        threshold=0.5,
        sim_type=None,
    ):

        check_matplotlib_support(f"{cls.__name__}.from_estimator")  # noqa
        import matplotlib.pyplot as plt  # noqa

        # set target_idx for multi-class estimators
        if hasattr(estimator, "classes_") and np.size(estimator.classes_) > 2:
            if target is None:
                raise ValueError("target must be specified for multi-class")
            target_idx = np.searchsorted(estimator.classes_, target)
            if (
                not (0 <= target_idx < len(estimator.classes_))
                or estimator.classes_[target_idx] != target
            ):
                raise ValueError("target not in est.classes_, got {}".format(target))
        else:
            # regression and binary classification
            target_idx = 0

        # Use check_array only on lists and other non-array-likes / sparse. Do not
        # convert DataFrame into a NumPy array.
        if not (hasattr(X, "__array__") or sparse.issparse(X)):
            X = check_array(X, force_all_finite="allow-nan", dtype=object)
        n_features = X.shape[1]

        feature_names = _check_feature_names(X, feature_names)
        # expand kind to always be a list of str
        kind_ = [kind] * len(features) if isinstance(kind, str) else kind
        if len(kind_) != len(features):
            raise ValueError(
                "When `kind` is provided as a list of strings, it should contain "
                f"as many elements as `features`. `kind` contains {len(kind_)} "
                f"element(s) and `features` contains {len(features)} element(s)."
            )

        # convert features into a seq of int tuples
        tmp_features, ice_for_two_way_pd = [], []
        for kind_plot, fxs in zip(kind_, features):
            if isinstance(fxs, (numbers.Integral, str)):
                fxs = (fxs,)
            try:
                fxs = tuple(
                    _get_feature_index(fx, feature_names=feature_names) for fx in fxs
                )
            except TypeError as e:
                raise ValueError(
                    "Each entry in features must be either an int, "
                    "a string, or an iterable of size at most 2."
                ) from e
            if not 1 <= np.size(fxs) <= 2:
                raise ValueError(
                    "Each entry in features must be either an int, "
                    "a string, or an iterable of size at most 2."
                )
            # store the information if 2-way PD was requested with ICE to later
            # raise a ValueError with an exhaustive list of problematic
            # settings.
            ice_for_two_way_pd.append(kind_plot != "average" and np.size(fxs) > 1)

            tmp_features.append(fxs)

        if any(ice_for_two_way_pd):
            # raise an error and be specific regarding the parameter values
            # when 1- and 2-way PD were requested
            kind_ = [
                "average" if forcing_average else kind_plot
                for forcing_average, kind_plot in zip(ice_for_two_way_pd, kind_)
            ]
            raise ValueError(
                "ICE plot cannot be rendered for 2-way feature interactions. "
                "2-way feature interactions mandates PD plots using the "
                "'average' kind: "
                f"features={features!r} should be configured to use "
                f"kind={kind_!r} explicitly."
            )
        features = tmp_features

        if categorical_features is None:
            is_categorical = [
                (False,) if len(fxs) == 1 else (False, False) for fxs in features
            ]
        else:
            # we need to create a boolean indicator of which features are
            # categorical from the categorical_features list.
            categorical_features = np.array(categorical_features, copy=False)
            if categorical_features.dtype.kind == "b":
                # categorical features provided as a list of boolean
                if categorical_features.size != n_features:
                    raise ValueError(
                        "When `categorical_features` is a boolean array-like, "
                        "the array should be of shape (n_features,). Got "
                        f"{categorical_features.size} elements while `X` contains "
                        f"{n_features} features."
                    )
                is_categorical = [
                    tuple(categorical_features[fx] for fx in fxs) for fxs in features
                ]
            elif categorical_features.dtype.kind in ("i", "O", "U"):
                # categorical features provided as a list of indices or feature names
                categorical_features_idx = [
                    _get_feature_index(cat, feature_names=feature_names)
                    for cat in categorical_features
                ]
                is_categorical = [
                    tuple([idx in categorical_features_idx for idx in fxs])
                    for fxs in features
                ]
            else:
                raise ValueError(
                    "Expected `categorical_features` to be an array-like of boolean,"
                    f" integer, or string. Got {categorical_features.dtype} instead."
                )

            for cats in is_categorical:
                if np.size(cats) == 2 and (cats[0] != cats[1]):
                    raise ValueError(
                        "Two-way partial dependence plots are not supported for pairs"
                        " of continuous and categorical features."
                    )

            # collect the indices of the categorical features targeted by the partial
            # dependence computation
            categorical_features_targeted = set(
                [
                    fx
                    for fxs, cats in zip(features, is_categorical)
                    for fx in fxs
                    if any(cats)
                ]
            )
            if categorical_features_targeted:
                min_n_cats = min(
                    [
                        len(_unique(_safe_indexing(X, idx, axis=1)))
                        for idx in categorical_features_targeted
                    ]
                )
                if grid_resolution < min_n_cats:
                    raise ValueError(
                        "The resolution of the computed grid is less than the "
                        "minimum number of categories in the targeted categorical "
                        "features. Expect the `grid_resolution` to be greater than "
                        f"{min_n_cats}. Got {grid_resolution} instead."
                    )

            for is_cat, kind_plot in zip(is_categorical, kind_):
                if any(is_cat) and kind_plot != "average":
                    raise ValueError(
                        "It is not possible to display individual effects for"
                        " categorical features."
                    )

        # Early exit if the axes does not have the correct number of axes
        if ax is not None and not isinstance(ax, plt.Axes):
            axes = np.asarray(ax, dtype=object)
            if axes.size != len(features):
                raise ValueError(
                    "Expected ax to have {} axes, got {}".format(
                        len(features), axes.size
                    )
                )

        for i in chain.from_iterable(features):
            if i >= len(feature_names):
                raise ValueError(
                    "All entries of features must be less than "
                    "len(feature_names) = {0}, got {1}.".format(len(feature_names), i)
                )

        if isinstance(subsample, numbers.Integral):
            if subsample <= 0:
                raise ValueError(
                    f"When an integer, subsample={subsample} should be positive."
                )
        elif isinstance(subsample, numbers.Real):
            if subsample <= 0 or subsample >= 1:
                raise ValueError(
                    f"When a floating-point, subsample={subsample} should be in "
                    "the (0, 1) range."
                )

        # compute predictions and/or averaged predictions
        pd_results = Parallel(n_jobs=n_jobs, verbose=verbose)(
            delayed(p_d)(
                estimator,
                X,
                fxs,
                feature_names=feature_names,
                categorical_features=categorical_features,
                response_method=response_method,
                method=method,
                grid_resolution=grid_resolution,
                percentiles=percentiles,
                kind=kind_plot,
                threshold=threshold,
                sim_type=sim_type,
            )
            for kind_plot, fxs in zip(kind_, features)
        )

        # For multioutput regression, we can only check the validity of target
        # now that we have the predictions.
        # Also note: as multiclass-multioutput classifiers are not supported,
        # multiclass and multioutput scenario are mutually exclusive. So there is
        # no risk of overwriting target_idx here.
        pd_result = pd_results[0]  # checking the first result is enough
        n_tasks = (
            pd_result.average.shape[0]
            if kind_[0] == "average"
            else pd_result.individual.shape[0]
        )
        if is_regressor(estimator) and n_tasks > 1:
            if target is None:
                raise ValueError("target must be specified for multi-output regressors")
            if not 0 <= target <= n_tasks:
                raise ValueError(
                    "target must be in [0, n_tasks], got {}.".format(target)
                )
            target_idx = target

        deciles = {}
        for fxs, cats in zip(features, is_categorical):
            for fx, cat in zip(fxs, cats):
                if not cat and fx not in deciles:
                    X_col = _safe_indexing(X, fx, axis=1)
                    deciles[fx] = mquantiles(X_col, prob=np.arange(0.1, 1.0, 0.1))

        display = pdpNew(
            pd_results=pd_results,
            features=features,
            feature_names=feature_names,
            target_idx=target_idx,
            deciles=deciles,
            kind=kind,
            subsample=subsample,
            random_state=random_state,
            is_categorical=is_categorical,
        )
        return display.plot(
            ax=ax,
            n_cols=n_cols,
            line_kw=line_kw,
            ice_lines_kw=ice_lines_kw,
            pd_line_kw=pd_line_kw,
            contour_kw=contour_kw,
            centered=centered,
        )


