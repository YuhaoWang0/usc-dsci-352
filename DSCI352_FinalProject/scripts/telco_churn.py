"""
Starter File – Telco Customer Churn (Keras + Sklearn + Optional AWS)

This is a starter scripts for your final project.

Your high-level goals:

1. Load and clean the Telco Customer Churn dataset.
2. Build a preprocessing pipeline that:
   - Handles numeric and categorical variables.
   - Imputes missing values.
   - Scales numeric features and one-hot encodes categorical features.
3. Train several ML models (scikit-learn + Keras):
   - Compare their performance (AUC, accuracy).
   - Interpret which model performs best and why.
4. (Bonus) Build a lightweight artifact suitable for deployment in AWS Lambda:
   - Export only the parameters needed to recreate the preprocessing + a simple model.
   - Upload this artifact to S3 if AWS export is enabled.

IMPORTANT:
- All places marked with "TODO (Student)" or "TODO" are for you to implement.
- As given, this file will raise NotImplementedError in several places.
  Your job is to replace those with working code. Disclaimer: this piece was AI generated.

Files:
- Input CSV: WA_Fn-UseC_-Telco-Customer-Churn.csv
"""

import os
import argparse
from dataclasses import dataclass
from pathlib import Path
import json
import joblib

import boto3
import numpy as np
import pandas as pd
import tensorflow as tf
from tensorflow import keras
from tensorflow.keras import layers
import matplotlib.pyplot as plt

from sklearn.model_selection import train_test_split
from sklearn.compose import ColumnTransformer
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import OneHotEncoder
from sklearn.impute import SimpleImputer
from sklearn.metrics import roc_auc_score, accuracy_score
from sklearn.linear_model import LogisticRegression, SGDClassifier
from sklearn.ensemble import RandomForestClassifier, GradientBoostingClassifier
from sklearn.preprocessing import StandardScaler

PROJECT_ROOT = Path(__file__).resolve().parents[1]
DATA_PATH = PROJECT_ROOT / "data" / "raw" / "WA_Fn-UseC_-Telco-Customer-Churn.csv"
OUTPUT_DIR = PROJECT_ROOT / "outputs" / "telco"

# AWS Config (for Bonus)
S3_BUCKET = os.environ.get("MODEL_BUCKET", "769341382710-model")
S3_KEY_LIGHT = os.environ.get("MODEL_KEY", "models/telco_churn_light.json")
AWS_REGION = os.environ.get("AWS_REGION", "us-east-1")

# Flag: make AWS completely optional
#   - False: run everything locally; no S3 upload
#   - True : in addition, build a lightweight artifact and upload to S3 (bonus)
ENABLE_AWS_EXPORT = False


# Data Loading and helpers (given to you)
def load_telco(path: str) -> pd.DataFrame:
    """
    Load the Telco CSV and perform basic cleaning:
      - Convert TotalCharges to numeric (coerce errors to NaN).
      - Convert Churn to 0/1.
      - Drop customerID (not useful as a feature).

    You can inspect and modify this function if you want, but it's provided
    as a starting point.
    """
    df = pd.read_csv(path)
    df["TotalCharges"] = pd.to_numeric(df["TotalCharges"], errors="coerce")
    df["Churn"] = df["Churn"].map({"Yes": 1, "No": 0})
    df = df.drop(columns=["customerID"], errors="ignore")
    return df


def build_preprocessor(df: pd.DataFrame):
    """
    Build a ColumnTransformer that:
      - Applies a numeric pipeline to numeric columns.
      - Applies a categorical pipeline to categorical columns.

    Numeric pipeline:
      - SimpleImputer(strategy="median")
      - StandardScaler()

    Categorical pipeline:
      - SimpleImputer(strategy="most_frequent")
      - OneHotEncoder(handle_unknown="ignore", sparse_output=False)

    Returns:
      preprocessor, X, y, numeric_feature_names, categorical_feature_names
    """
    X = df.drop("Churn", axis=1)
    y = df["Churn"]

    numeric_features = X.select_dtypes(include=["int64", "float64"]).columns.tolist()
    categorical_features = X.select_dtypes(include=["object", "bool"]).columns.tolist()

    numeric_transformer = Pipeline(
        steps=[
            ("imputer", SimpleImputer(strategy="median")),
            ("scaler", StandardScaler()),
        ]
    )
    categorical_transformer = Pipeline(
        steps=[
            ("imputer", SimpleImputer(strategy="most_frequent")),
            ("onehot", OneHotEncoder(handle_unknown="ignore", sparse_output=False)),
        ]
    )

    preprocessor = ColumnTransformer(
        transformers=[
            ("num", numeric_transformer, numeric_features),
            ("cat", categorical_transformer, categorical_features),
        ]
    )
    return preprocessor, X, y, numeric_features, categorical_features


def make_numpy_for_keras(preprocessor, X_train, X_val):
    """
    Fit the preprocessor on X_train, transform train+val, and return dense
    numpy arrays so that Keras can use them.

    Also returns feature_names for inspection (not required for training).
    """
    preprocessor.fit(X_train)
    X_train_proc = preprocessor.transform(X_train)
    X_val_proc = preprocessor.transform(X_val)

    ohe = preprocessor.named_transformers_["cat"].named_steps["onehot"]
    cat_names = ohe.get_feature_names_out()
    num_names = preprocessor.transformers_[0][2]
    feature_names = np.concatenate([num_names, cat_names])

    return X_train_proc.astype("float32"), X_val_proc.astype("float32"), feature_names


class GradientTracker(keras.callbacks.Callback):
    """
    Tracks gradient statistics after each epoch for a Keras model.

    This helps you reason about gradient health
    (e.g. vanishing/exploding gradients).
    """

    def __init__(self, model, sample_x, sample_y):
        super().__init__()
        self.model_for_grads = model
        self.sample_x = tf.convert_to_tensor(sample_x)
        self.sample_y = tf.convert_to_tensor(sample_y, dtype=tf.float32)
        self.history = []

    def on_epoch_end(self, epoch, logs=None):
        with tf.GradientTape() as tape:
            preds = self.model_for_grads(self.sample_x, training=True)  # (batch, 1)
            preds = tf.squeeze(preds, axis=-1)  # -> (batch,)
            loss = keras.losses.binary_crossentropy(self.sample_y, preds)
            loss = tf.reduce_mean(loss)

        grads = tape.gradient(loss, self.model_for_grads.trainable_weights)
        grad_vals = [tf.reshape(g, [-1]) for g in grads if g is not None]
        if grad_vals:
            all_grads = tf.concat(grad_vals, axis=0)
            grad_mean = tf.reduce_mean(tf.abs(all_grads)).numpy().item()
            grad_max = tf.reduce_max(tf.abs(all_grads)).numpy().item()
            grad_std = tf.math.reduce_std(all_grads).numpy().item()
        else:
            grad_mean = grad_max = grad_std = 0.0

        self.history.append(
            {
                "epoch": int(epoch),
                "grad_abs_mean": grad_mean,
                "grad_abs_max": grad_max,
                "grad_std": grad_std,
                "loss": float(loss.numpy()),
                "logs": logs or {},
            }
        )


# ##################
# Keras Model – TODO
def build_keras_model(input_dim: int):
    """
    TODO:
    Build and compile a Keras model that takes as input a vector of length `input_dim`
    (the preprocessed features) and predicts churn probability.

    Requirements / suggestions:
      - Use a Sequential model starting with layers.Input(shape=(input_dim,))
      - Add several Dense layers with ReLU activation, optionally with Dropout.
      - Final layer: Dense(1, activation="sigmoid") (binary classification).
      - Compile with:
          optimizer = Adam(learning_rate=1e-3) (or similar)
          loss      = "binary_crossentropy"
          metrics   = [AUC, "accuracy"]

    Return:
      - a compiled Keras model ready for training.
    """
    model = keras.Sequential([
        layers.Input(shape = (input_dim, )),
        layers.Dense(128, activation = 'relu'),
        layers.Dropout(0.3),
        layers.Dense(64, activation = 'relu'),
        layers.Dropout(0.3),
        layers.Dense(32, activation = 'relu'),
        layers.Dense(1, activation = 'sigmoid')
    ])

    model.compile(
        optimizer = keras.optimizers.Adam(learning_rate = 1e-3),
        loss = 'binary_crossentropy',
        metrics = [keras.metrics.AUC(name = 'auc'), 'accuracy']
    )

    return model

############################
# Scikit-Learn Models – TODO
def train_sklearn_models(preprocessor, X, y):
    """
    TODO (Student):
    1. Split X, y into train and validation sets:
         X_train, X_val, y_train, y_val = train_test_split(
             X, y, test_size=0.2, random_state=42, stratify=y
         )

    2. Build and train several scikit-learn models using a Pipeline that
       includes the `preprocessor` plus a classifier, for example:
         - LogisticRegression(max_iter=500)
         - RandomForestClassifier(...)
         - GradientBoostingClassifier(...)
         - SGDClassifier(loss="log_loss", ...)

       For each model:
         - Fit on X_train, y_train.
         - Use predict_proba on X_val to get positive-class probabilities.
         - Compute validation AUC via roc_auc_score(y_val, proba).
         - Store results in a list as tuples:
               ("model_name_string", trained_pipeline, auc_value)

    3. Sort the results list by AUC (descending), so that results[0]
       is your best sklearn model.

    4. For the AWS lightweight artifact, pick ONE "simple" model that
       is Lambda-friendly (for example, an SGDClassifier with log_loss).
       Return:
         results, (X_train, X_val, y_train, y_val), chosen_sgd_pipeline, auc_of_sgd

    Returns:
      - results: list of (name, model_pipeline, auc)
      - splits: (X_train, X_val, y_train, y_val)
      - sgd_model: the chosen SGD-based pipeline for deployment
      - sgd_auc: AUC of that SGD model
    """

    X_train, X_val, y_train, y_val = train_test_split(
        X, y, test_size = 0.2, random_state = 42, stratify = y
    )

    models = {
        'Logistic Regression': LogisticRegression(max_iter = 500),
        'Random Forest': RandomForestClassifier(n_estimators = 100, random_state = 42),
        'Gradient Boosting': GradientBoostingClassifier(random_state = 42),
        'SGDClassifier': SGDClassifier(loss = 'log_loss', max_iter = 1000, random_state = 42)
    }

    results = []
    sgd_model = None
    sgd_auc = None
    for name, clf in models.items():
        pipeline = Pipeline([
            ('preprocessor', preprocessor),
            ('clf', clf)
        ])
        pipeline.fit(X_train, y_train)
        y_val_proba = pipeline.predict_proba(X_val)[:, 1]
        auc = roc_auc_score(y_val, y_val_proba)

        results.append((name, pipeline, auc))

        if name == 'SGDClassifier':
            sgd_model = pipeline
            sgd_auc = auc

    results.sort(key = lambda x: x[2], reverse = True)
    return results, (X_train, X_val, y_train, y_val), sgd_model, sgd_auc

def plot_model_comparison(model_leaderboard, outpath):
    """
    Plot AUC comparison for all models in the leaderboard.
    """
    names = [item["name"] for item in model_leaderboard]
    aucs = [item["auc"] for item in model_leaderboard]

    plt.figure(figsize=(10, 5))
    bars = plt.bar(names, aucs)
    plt.ylabel("AUC")
    plt.title("Telco Model Comparison by AUC")
    plt.xticks(rotation=25, ha="right")
    plt.ylim(0, 1)

    # 在每个柱子上标数值
    for bar, auc in zip(bars, aucs):
        plt.text(
            bar.get_x() + bar.get_width() / 2,
            bar.get_height() + 0.005,
            f"{auc:.4f}",
            ha="center",
            va="bottom",
            fontsize=9
        )

    plt.tight_layout()
    plt.savefig(outpath, dpi=150)
    plt.close()

################################################
# 4) Lightweight Artifact for AWS – TODO (Bonus)
def build_lightweight_artifact(
    df,
    preprocessor,
    numeric_features,
    categorical_features,
    sgd_model,
):
    """
    TODO (Bonus):
    Build a lightweight artifact (a plain Python dict) that contains JUST enough
    information to reconstruct the preprocessing + SGD classifier inside
    an AWS Lambda function (without sklearn).

    Suggested steps:

    1. Fit `preprocessor` on the full feature set X_full (df.drop("Churn", axis=1)).
       This ensures the imputer, scaler, and OneHotEncoder see all categories.

    2. Extract numeric imputer and scaler parameters:
         num_pipe = preprocessor.named_transformers_["num"]
         imputer  = num_pipe.named_steps["imputer"]
         scaler   = num_pipe.named_steps["scaler"]

       Save:
         - numeric_medians = imputer.statistics_.tolist()
         - numeric_means   = scaler.mean_.tolist()
         - numeric_scales  = scaler.scale_.tolist()

    3. Extract categorical one-hot encoder info:
         cat_pipe = preprocessor.named_transformers_["cat"]
         ohe      = cat_pipe.named_steps["onehot"]

       Save:
         - categories for each categorical feature: [c.tolist() for c in ohe.categories_]

    4. Extract SGD classifier coefficients and intercept:
         clf = sgd_model.named_steps["clf"]
         coef = clf.coef_.tolist()
         intercept = clf.intercept_.tolist()

    5. Construct and return a dict like:

       artifact = {
         "numeric_features": numeric_features,
         "numeric_imputer_medians": numeric_medians,
         "numeric_means": numeric_means,
         "numeric_scales": numeric_scales,
         "categorical_features": categorical_features,
         "categories": [...],
         "coef": coef,
         "intercept": intercept,
       }

    This dict should be fully JSON-serializable.
    """
    X_full = df.drop('Churn', axis = 1)
    preprocessor.fit(X_full)

    num_pipe = preprocessor.named_transformers_['num']
    imputer = num_pipe.named_steps['imputer']
    scaler = num_pipe.named_steps['scaler']

    numeric_medians = imputer.statistics_.tolist()
    numeric_means = scaler.mean_.tolist()
    numeric_scales = scaler.scale_.tolist()

    cat_pipe = preprocessor.named_transformers_['cat']
    ohe = cat_pipe.named_steps['onehot']
    categories = [c.tolist() for c in ohe.categories_]

    clf = sgd_model.named_steps['clf']
    coef = clf.coef_.tolist()
    intercept = clf.intercept_.tolist()
    classes = clf.classes_.tolist()

    artifact = {
        'numeric_features': list(numeric_features),
        'numeric_imputer_medians': numeric_medians,
        'numeric_means': numeric_means,
        'numeric_scales': numeric_scales,
        'categorical_features': list(categorical_features),
        'categories': categories,
        'coef': coef,
        'intercept': intercept,
        'classes': classes
    }
    return artifact



#######################
# AWS Upload (given)
def upload_json_to_s3(obj, bucket, key, region="us-east-1"):
    """
    Upload a JSON-serializable object to S3. Used only if ENABLE_AWS_EXPORT is True.
    """
    s3 = boto3.client("s3", region_name=region)
    body = json.dumps(obj).encode("utf-8")
    s3.put_object(
        Bucket=bucket,
        Key=key,
        Body=body,
        ContentType="application/json",
    )
    print(f"Uploaded to s3://{bucket}/{key}")

def clean_telco_input(df: pd.DataFrame) -> pd.DataFrame:
    """
    Clean new input data for prediction.
    """
    df = df.copy()
    df["TotalCharges"] = pd.to_numeric(df["TotalCharges"], errors="coerce")
    df = df.drop(columns=["customerID"], errors="ignore")
    df = df.drop(columns=["Churn"], errors="ignore")
    return df

def main():
    parser = argparse.ArgumentParser()
    parser.add_argument(
        "--input",
        type=str,
        default=str(DATA_PATH),
        help="Path to Telco CSV file"
    )
    parser.add_argument(
        "--outdir",
        type=str,
        default=str(OUTPUT_DIR),
        help="Output folder"
    )
    args = parser.parse_args()

    outdir = Path(args.outdir)
    outdir.mkdir(parents=True, exist_ok=True)

    # Step 0: Load data
    if not os.path.exists(args.input):
        raise FileNotFoundError(f"CSV not found at {args.input}")

    print("Loading telco data...")
    df = load_telco(args.input)

    # Step 1: Preprocessor
    print("Building preprocessor...")
    preprocessor, X, y, num_feats, cat_feats = build_preprocessor(df)

    # Step 2: Scikit-Learn models
    print("Training sklearn models...")
    sk_results, splits, sgd_model, sgd_auc = train_sklearn_models(preprocessor, X, y)
    X_train, X_val, y_train, y_val = splits

    joblib.dump(sgd_model, outdir / "telco_pipeline.pkl")
    print(f"Saved model to {outdir / 'telco_pipeline.pkl'}")

    best_sklearn_name, best_sklearn_model, best_sklearn_auc = sk_results[0]

    # Step 3: Keras model
    print("Preparing numpy for Keras...")
    X_train_np, X_val_np, feature_names = make_numpy_for_keras(preprocessor, X_train, X_val)

    print("Building Keras model...")
    keras_model = build_keras_model(input_dim=X_train_np.shape[1])

    sample_x = X_train_np[:256]
    sample_y = y_train.values[:256]
    grad_tracker = GradientTracker(keras_model, sample_x, sample_y)

    print("Training Keras model...")
    history = keras_model.fit(
        X_train_np,
        y_train.values,
        validation_data=(X_val_np, y_val.values),
        epochs=20,
        batch_size=256,
        callbacks=[grad_tracker],
        verbose=1,
    )

    # Step 4: Save gradient history
    with open(outdir / "keras_gradients.json", "w") as f:
        json.dump(grad_tracker.history, f, indent=2)
    print(f"Saved gradient stats to {outdir / 'keras_gradients.json'}")

    # Step 5: Evaluate Keras on validation set
    y_val_pred = keras_model.predict(X_val_np).ravel()
    keras_auc = roc_auc_score(y_val.values, y_val_pred)
    keras_acc = accuracy_score(y_val.values, (y_val_pred >= 0.5).astype(int))

    last_grad = grad_tracker.history[-1]
    grad_quality = {
        "last_epoch_grad_abs_mean": last_grad["grad_abs_mean"],
        "last_epoch_grad_abs_max": last_grad["grad_abs_max"],
        "last_epoch_grad_std": last_grad["grad_std"],
    }

    print("\n========= MODEL SUMMARY =========")
    print(f"Best sklearn: {best_sklearn_name} AUC={best_sklearn_auc:.4f}")
    print(f"Keras model : AUC={keras_auc:.4f} ACC={keras_acc:.4f}")
    print(f"SGD (for potential Lambda): AUC={sgd_auc:.4f}")
    print(f"Gradient quality (keras): {grad_quality}")

    # Step 6: Choose final model
    if keras_auc >= best_sklearn_auc:
        final_model_name = "keras_model"
        final_auc = keras_auc
    else:
        final_model_name = best_sklearn_name
        final_auc = best_sklearn_auc

    print(f"\nFINAL CHOICE for my Report: {final_model_name} AUC={final_auc:.4f}")

    # Step 7: Build leaderboard
    model_leaderboard = []
    for name, _, auc_val in sk_results:
        model_leaderboard.append(
            {
                "name": name,
                "type": "sklearn",
                "auc": float(auc_val),
            }
        )

    model_leaderboard.append(
        {
            "name": "keras_mlp",
            "type": "keras",
            "auc": float(keras_auc),
            "accuracy": float(keras_acc),
            "grad_abs_mean": float(grad_tracker.history[-1]["grad_abs_mean"]),
            "grad_abs_max": float(grad_tracker.history[-1]["grad_abs_max"]),
            "grad_std": float(grad_tracker.history[-1]["grad_std"]),
        }
    )

    # Save plot, save JSON
    plot_model_comparison(model_leaderboard, outdir / "telco_auc_comparison.png")
    print(f"Saved {outdir / 'telco_auc_comparison.png'}")

    with open(outdir / "model_leaderboard_telco.json", "w") as f:
        json.dump(model_leaderboard, f, indent=2)
    print(f"Saved {outdir / 'model_leaderboard_telco.json'}")

    # Step 8: Bonus AWS
    if ENABLE_AWS_EXPORT:
        print("\nENABLE_AWS_EXPORT=True → building lightweight artifact and uploading to S3.")

        artifact = build_lightweight_artifact(
            df,
            preprocessor,
            num_feats,
            cat_feats,
            sgd_model,
        )

        wrapped_artifact = {
            "artifact": artifact,
            "training_meta": {
                "final_choice": final_model_name,
                "final_choice_auc": float(final_auc),
                "best_sklearn": best_sklearn_name,
                "best_sklearn_auc": float(best_sklearn_auc),
                "keras_auc": float(keras_auc),
                "keras_acc": float(keras_acc),
                "model_leaderboard": model_leaderboard,
            },
        }

        upload_json_to_s3(wrapped_artifact, S3_BUCKET, S3_KEY_LIGHT, AWS_REGION)
        print("Uploaded lightweight artifact with metadata to S3.")

if __name__ == "__main__":
    main()
