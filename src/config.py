"""Configuration file for translation evaluation metrics."""

# Reference-free metrics
REFERENCE_FREE_METRICS = {
    "cometkiwi-xxl": "Unbabel/wmt23-cometkiwi-da-xxl",
    "cometkiwi-xl": "Unbabel/wmt23-cometkiwi-da-xl",
    "xcomet": "Unbabel/XCOMET-XXL",
    "cometkiwi-22": "Unbabel/wmt22-cometkiwi-da",
}

# Reference-based metrics
REFERENCE_BASED_METRICS = {
    "comet-22": "Unbabel/wmt22-comet-da",
    "xcomet": "Unbabel/XCOMET-XXL",
}

# Metrics that can be used in both scenarios
DUAL_MODE_METRICS = {
    "xcomet": "Unbabel/XCOMET-XXL",
}

METRICX_MODELS = {
    # MetricX-24 (hybrid - both reference-based and QE)
    "metricx-24-hybrid-xxl": "google/metricx-24-hybrid-xxl-v2p6",
    "metricx-24-hybrid-xl": "google/metricx-24-hybrid-xl-v2p6",
    "metricx-24-hybrid-large": "google/metricx-24-hybrid-large-v2p6",
    "metricx-24-hybrid-xxl-bf16": "google/metricx-24-hybrid-xxl-v2p6-bfloat16",
    "metricx-24-hybrid-xl-bf16": "google/metricx-24-hybrid-xl-v2p6-bfloat16",
    "metricx-24-hybrid-large-bf16": "google/metricx-24-hybrid-large-v2p6-bfloat16",
    # MetricX-23 (reference-based)
    "metricx-23-xxl": "google/metricx-23-xxl-v2p0",
    "metricx-23-xl": "google/metricx-23-xl-v2p0",
    "metricx-23-large": "google/metricx-23-large-v2p0",
    # MetricX-23 (QE - reference-free)
    "metricx-23-qe-xxl": "google/metricx-23-qe-xxl-v2p0",
    "metricx-23-qe-xl": "google/metricx-23-qe-xl-v2p0",
    "metricx-23-qe-large": "google/metricx-23-qe-large-v2p0",
}

# Default configuration
DEFAULT_CONFIG = {
    "source_lang": "en",
    "target_lang": "uk",
    "batch_size": 8,
    "device": "cuda",
    "output_dir": "results",
    "cache_dir": ".cache",
}
