# Dataset Builder

## Architecture

`DatasetBuilder` is AlphaForge's only interface between Feature Engineering and
Machine Learning. It accepts an engineered `pandas.DataFrame` and returns a
`DatasetBundle` containing chronological training, validation, and test data.
It does not download data, train models, save models, or produce plots.

## Target Generation

The default target is `NEXT_DAY_DIRECTION`. It is one when the next session's
close is greater than the current close and zero otherwise. It uses
`Close.shift(-1)` and removes the final row, because that row has no future
close from which to construct a valid label.

`TargetBuilder` keeps named target strategies in one location so future
regression targets can be added without changing the dataset-building flow.

## Split Strategy

`ChronologicalSplitter` never shuffles data. The default allocation is 70% for
training, 15% for validation, and 15% for testing. Earlier observations always
belong to the training partition, followed by validation and then testing.
This preserves temporal causality in market data.

## Data Leakage Prevention

Missing rows are removed before target creation. The target is created using
only the next close, and the final unusable row is removed. Feature scaling is
fit exclusively on the training partition; validation and test partitions are
only transformed with training statistics. No validation or test observation
influences scaling parameters.

## Scaling Strategy

Scaling is disabled by default. When `scale=True`, `FeatureScaler` performs
standard scaling with the training mean and population standard deviation.
Constant training features use a scale of one so they remain finite. The scaler
is intentionally isolated so additional strategies, such as MinMax scaling,
can be added without changing `DatasetBuilder`.

## DatasetBundle

`DatasetBundle` is a typed dataclass containing feature matrices and target
series for all three splits, feature and target names, split boundaries, and
metadata. Metadata records the symbol, final row count, feature count, target,
split ratios, scaling state, and UTC creation timestamp.
