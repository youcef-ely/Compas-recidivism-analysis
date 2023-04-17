from aif360.metrics import BinaryLabelDatasetMetric


def compute_metrics(dataset, unprivileged_groups: list, privileged_groups: list):
    binary_metric_orig_train = BinaryLabelDatasetMetric(dataset,
                        unprivileged_groups = unprivileged_groups,
                        privileged_groups = privileged_groups)

    mean_diff = binary_metric_orig_train.statistical_parity_difference()
    smoothed_empirical = binary_metric_orig_train.smoothed_empirical_differential_fairness()
    disparate_impact = binary_metric_orig_train.disparate_impact()
    return mean_diff, smoothed_empirical, disparate_impact