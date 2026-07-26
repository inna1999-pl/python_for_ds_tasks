from typing import List, Optional, Tuple

import pandas as pd

from sklearn.model_selection import train_test_split
from sklearn.preprocessing import MinMaxScaler, OneHotEncoder


TARGET_COL = 'Exited'

INPUT_COLS = [
    'CreditScore',
    'Geography',
    'Gender',
    'Age',
    'Tenure',
    'Balance',
    'NumOfProducts',
    'HasCrCard',
    'IsActiveMember',
    'EstimatedSalary'
]

NUMERIC_COLS = [
    'CreditScore',
    'Age',
    'Tenure',
    'Balance',
    'NumOfProducts',
    'HasCrCard',
    'IsActiveMember',
    'EstimatedSalary'
]

CATEGORICAL_COLS = [
    'Geography',
    'Gender'
]


def split_data(
    raw_df: pd.DataFrame,
    input_cols: List[str],
    target_col: str = TARGET_COL,
    test_size: float = 0.2,
    random_state: int = 42
) -> Tuple[pd.DataFrame, pd.DataFrame, pd.Series, pd.Series]:
    """
    Розділяє сирі дані на тренувальну та валідаційну вибірки.
    """

    X = raw_df[input_cols].copy()
    y = raw_df[target_col].copy()

    X_train, X_val, train_targets, val_targets = train_test_split(
        X,
        y,
        test_size=test_size,
        random_state=random_state,
        stratify=y
    )

    return X_train, X_val, train_targets, val_targets


def fit_encoder(
    X_train: pd.DataFrame,
    categorical_cols: List[str]
) -> OneHotEncoder:
    """
    Навчає OneHotEncoder на категоріальних колонках
    тренувальної вибірки.
    """

    encoder = OneHotEncoder(
        handle_unknown='ignore',
        sparse_output=False
    )

    encoder.fit(X_train[categorical_cols])

    return encoder


def encode_categorical_data(
    df: pd.DataFrame,
    encoder: OneHotEncoder,
    categorical_cols: List[str]
) -> pd.DataFrame:
    """
    Кодує категоріальні ознаки за допомогою
    вже навченого OneHotEncoder.
    """

    encoded_values = encoder.transform(
        df[categorical_cols]
    )

    encoded_columns = encoder.get_feature_names_out(
        categorical_cols
    )

    encoded_df = pd.DataFrame(
        encoded_values,
        columns=encoded_columns,
        index=df.index
    )

    return encoded_df


def fit_scaler(
    X_train: pd.DataFrame,
    numeric_cols: List[str]
) -> MinMaxScaler:
    """
    Навчає MinMaxScaler на числових колонках
    тренувальної вибірки.
    """

    scaler = MinMaxScaler()

    scaler.fit(X_train[numeric_cols])

    return scaler


def scale_numeric_data(
    df: pd.DataFrame,
    scaler: MinMaxScaler,
    numeric_cols: List[str]
) -> pd.DataFrame:
    """
    Масштабує числові ознаки за допомогою
    вже навченого scaler.
    """

    scaled_values = scaler.transform(
        df[numeric_cols]
    )

    scaled_df = pd.DataFrame(
        scaled_values,
        columns=numeric_cols,
        index=df.index
    )

    return scaled_df


def combine_features(
    numeric_df: pd.DataFrame,
    categorical_df: pd.DataFrame
) -> pd.DataFrame:
    """
    Об'єднує числові та закодовані категоріальні ознаки.
    """

    result = pd.concat(
        [
            numeric_df.reset_index(drop=True),
            categorical_df.reset_index(drop=True)
        ],
        axis=1
    )

    return result


def preprocess_data(
    raw_df: pd.DataFrame,
    scaler_numeric: bool = True,
    test_size: float = 0.2,
    random_state: int = 42
) -> Tuple[
    pd.DataFrame,
    pd.Series,
    pd.DataFrame,
    pd.Series,
    List[str],
    Optional[MinMaxScaler],
    OneHotEncoder
]:
    """
    Виконує повну обробку тренувальних даних.
    """

    input_cols = INPUT_COLS.copy()

    X_train_raw, X_val_raw, train_targets, val_targets = split_data(
        raw_df=raw_df,
        input_cols=input_cols,
        target_col=TARGET_COL,
        test_size=test_size,
        random_state=random_state
    )

    encoder = fit_encoder(
        X_train=X_train_raw,
        categorical_cols=CATEGORICAL_COLS
    )

    X_train_categorical = encode_categorical_data(
        df=X_train_raw,
        encoder=encoder,
        categorical_cols=CATEGORICAL_COLS
    )

    X_val_categorical = encode_categorical_data(
        df=X_val_raw,
        encoder=encoder,
        categorical_cols=CATEGORICAL_COLS
    )

    if scaler_numeric:
        scaler = fit_scaler(
            X_train=X_train_raw,
            numeric_cols=NUMERIC_COLS
        )

        X_train_numeric = scale_numeric_data(
            df=X_train_raw,
            scaler=scaler,
            numeric_cols=NUMERIC_COLS
        )

        X_val_numeric = scale_numeric_data(
            df=X_val_raw,
            scaler=scaler,
            numeric_cols=NUMERIC_COLS
        )

    else:
        scaler = None

        X_train_numeric = X_train_raw[
            NUMERIC_COLS
        ].copy()

        X_val_numeric = X_val_raw[
            NUMERIC_COLS
        ].copy()

    X_train = combine_features(
        numeric_df=X_train_numeric,
        categorical_df=X_train_categorical
    )

    X_val = combine_features(
        numeric_df=X_val_numeric,
        categorical_df=X_val_categorical
    )

    train_targets = train_targets.reset_index(drop=True)
    val_targets = val_targets.reset_index(drop=True)

    return (
        X_train,
        train_targets,
        X_val,
        val_targets,
        input_cols,
        scaler,
        encoder
    )


def preprocess_new_data(
    new_df: pd.DataFrame,
    input_cols: List[str],
    scaler: Optional[MinMaxScaler],
    encoder: OneHotEncoder
) -> pd.DataFrame:
    """
    Обробляє нові дані за допомогою вже навчених
    scaler та encoder.
    """

    new_inputs = new_df[input_cols].copy()

    categorical_df = encode_categorical_data(
        df=new_inputs,
        encoder=encoder,
        categorical_cols=CATEGORICAL_COLS
    )

    if scaler is not None:
        numeric_df = scale_numeric_data(
            df=new_inputs,
            scaler=scaler,
            numeric_cols=NUMERIC_COLS
        )

    else:
        numeric_df = new_inputs[
            NUMERIC_COLS
        ].copy()

    processed_df = combine_features(
        numeric_df=numeric_df,
        categorical_df=categorical_df
    )

    return processed_df