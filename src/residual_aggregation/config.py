"""Shared constants for the residual aggregation experiments."""

DATE = "date"
BRAND = "brand"
TARGET = "sales"
MARKET_SIZE = "market_size"
RESIDUAL = "Residual"
RANDOM_SEED = 20260623

BRANDS = [
    "Audi",
    "BMW",
    "Ford",
    "Hyundai",
    "Kia",
    "Mazda",
    "Mercedes-Benz",
    "Nissan",
    "Peugeot",
    "Skoda",
    "Tesla",
    "Toyota",
    "Volkswagen",
    "Volvo",
]

K_VALUES = (5, 8, 10, 12, 14)
TEST_START_INDEX = 48
BOOTSTRAP_REPS = 10000
N_PLACEBO_REPLICATES = 20

LGB_PARAMS = {
    "objective": "regression_l1",
    "metric": "mae",
    "learning_rate": 0.03,
    "num_leaves": 7,
    "max_depth": 3,
    "min_data_in_leaf": 20,
    "lambda_l1": 0.1,
    "lambda_l2": 1.0,
    "feature_fraction": 0.9,
    "bagging_fraction": 0.9,
    "bagging_freq": 1,
    "seed": RANDOM_SEED,
    "deterministic": True,
    "force_col_wise": True,
    "num_threads": 1,
    "verbosity": -1,
}
