# -*- coding: utf-8 -*-
"""
Created on Wed Nov 18 15:16:54 2026

@author: 28190
"""

# -*- coding: utf-8 -*-
# Active Chlorine Reactivity Prediction Platform
#
# Repository layout:
# app.py
# requirements.txt
# packages.txt
# data/
#   DFT_descriptor_library.csv
#   FreeChlorine_second_DFT_13features_records.csv
#   ChlorineRadical_second_DFT_13features_records.csv
#   DichlorineRadical_second_DFT_13features_records.csv

import csv
import hashlib
import html
import io
import json
import math
import os
import socket
import subprocess
import sys
import threading
import time
import unicodedata
import webbrowser

from functools import lru_cache
from pathlib import Path
from urllib.parse import quote
from urllib.request import Request, urlopen

import numpy as np
import pandas as pd
import streamlit as st

from rdkit import Chem, RDLogger
from rdkit.Chem import Descriptors, Draw
from rdkit.Chem.MolStandardize import rdMolStandardize as MS

from sklearn.impute import SimpleImputer

from streamlit.runtime.scriptrunner import get_script_run_ctx


# ============================================================
# 1. Runtime and page configuration
# ============================================================

IN_STREAMLIT = (
    get_script_run_ctx(suppress_warning=True) is not None
)

if IN_STREAMLIT:
    st.set_page_config(
        page_title="Active Chlorine Reactivity Predictor",
        page_icon="🧪",
        layout="wide",
        initial_sidebar_state="collapsed",
    )

RDLogger.DisableLog("rdApp.*")


# ============================================================
# 2. Repository-relative paths
# ============================================================

BASE = Path(__file__).resolve().parent
DATA = BASE / "data"
LIBRARY = DATA / "DFT_descriptor_library.csv"


# ============================================================
# 3. Feature definitions
# ============================================================

STRUCT = [
    "MolLogP",
    "TPSA",
    "MolMR",
    "BalabanJ",
    "SlogP_VSA6",
    "NumHAcceptors",
    "NumHDonors",
    "NHOHCount",
    "RingCount",
    "NumAromaticRings",
    "FractionCSP3",
    "Kappa2",
    "fr_C_O",
    "fr_phenol",
    "fr_halogen",
]

DFT = [
    "EHOMO(N)",
    "ELUMO(N)",
    "Egap",
    "Vertical IP",
    "Vertical EA",
    "Mulliken electronegativity χ",
    "Chemical potential μ",
    "Hardness η",
    "Softness S",
    "Electrophilicity index ω",
    "f−(max)",
    "f+(max)",
    "f0(max)",
]

# Alternative ASCII column names accepted in uploaded CSV files.
ASCII_DFT = [
    "EHOMO_N_eV",
    "ELUMO_N_eV",
    "Egap_eV",
    "Vertical_IP_eV",
    "Vertical_EA_eV",
    "Mulliken_electronegativity_chi_eV",
    "Chemical_potential_mu_eV",
    "Hardness_eta_eV",
    "Softness_S_eV_inv",
    "Electrophilicity_index_omega_eV",
    "f_minus_max",
    "f_plus_max",
    "f_zero_max",
]


# ============================================================
# 4. Selected model configurations
# ============================================================

CONFIG = {
    "Free chlorine": {
        "file": "FreeChlorine_second_DFT_13features_records.csv",
        "tag": "FreeChlorine_project",
        "algorithm": "XGB",
        "params": dict(
            n_estimators=306,
            max_depth=5,
            learning_rate=0.03485638237861316,
            subsample=0.4,
            colsample_bytree=0.969846712159922,
            min_child_weight=7.3588090746782475,
            gamma=0.6475713918482008,
            reg_alpha=6.7200389956947145e-06,
            reg_lambda=0.001,
        ),
    },

    "Chlorine radical": {
        "file": "ChlorineRadical_second_DFT_13features_records.csv",
        "tag": "ChlorineRadical_project",
        "algorithm": "XGB",
        "params": dict(
            n_estimators=1000,
            max_depth=4,
            learning_rate=0.027810503145741048,
            subsample=0.7727791037409364,
            colsample_bytree=0.6092047449230159,
            min_child_weight=15.571886888557723,
            gamma=0.6441872361006464,
            reg_alpha=0.00511266061212155,
            reg_lambda=1.9846523877112163,
        ),
    },

    "Dichlorine radical": {
        "file": "DichlorineRadical_second_DFT_13features_records.csv",
        "tag": "DichlorineRadical_project",
        "algorithm": "CatBoost",
        "params": dict(
            iterations=1302,
            depth=5,
            learning_rate=0.02272714608183828,
            l2_leaf_reg=0.17385615157119938,
            random_strength=8.543960310861761,
            bagging_temperature=1.0,
            border_count=243,
            min_data_in_leaf=25,
        ),
    },
}

ROUTES = [
    "Library DFT",
    "Upload DFT CSV",
    "Structure-only",
]

# Used only for parent-structure identity matching.
INERT = {
    "O",
    "F",
    "Cl",
    "Br",
    "I",
    "[F-]",
    "[Cl-]",
    "[Br-]",
    "[I-]",
    "[Li+]",
    "[Na+]",
    "[K+]",
    "[Mg+2]",
    "[Ca+2]",
    "O=S(=O)(O)O",
    "O=S(=O)([O-])[O-]",
    "O=S(=O)([O-])O",
}


# ============================================================
# 5. Interface styling
# ============================================================

CSS = """
<style>
.stApp {
    background: #f5f7fb;
}

.block-container {
    max-width: 1260px;
    padding: 1.5rem 2rem;
}

.stApp,
.stApp input,
.stApp button {
    font-family: Arial, sans-serif;
}

.app-banner {
    background: #123c75;
    color: white;
    border-radius: 12px;
    padding: 22px;
    margin-bottom: 20px;
    text-align: center;
    font-size: 26px;
    font-weight: 700;
}

h3 {
    color: #123c75 !important;
    font-size: 22px !important;
}

[data-testid="stVerticalBlockBorderWrapper"] {
    background: white;
    border-radius: 12px;
}

[data-testid="stMetric"] {
    background: #eef4fc;
    border-left: 5px solid #174d8e;
    border-radius: 8px;
    padding: 16px 22px;
    margin-top: 8px;
}

[data-testid="stMetricValue"] {
    color: #cf3039;
    font-size: 34px;
}

.compound-name {
    color: #123c75;
    font-weight: 700;
    font-size: 19px;
}

.empty-panel {
    color: #718198;
    padding: 36px 12px;
    text-align: center;
}

@media(max-width: 800px) {
    .block-container {
        padding: 1rem;
    }

    .app-banner {
        font-size: 21px;
    }
}
</style>
"""


# ============================================================
# 6. General utilities
# ============================================================

def cached(kind, **kwargs):
    if IN_STREAMLIT:
        return getattr(st, kind)(**kwargs)

    return lambda function: function


def norm(value):
    return " ".join(
        unicodedata.normalize("NFKC", str(value))
        .casefold()
        .split()
    )


def digest(value):
    raw = (
        value
        if isinstance(value, bytes)
        else repr(value).encode("utf-8")
    )

    return hashlib.sha256(raw).hexdigest()[:24]


def file_stamp(path):
    if not path.is_file():
        raise FileNotFoundError(
            "Missing repository data file: data/" + path.name
        )

    info = path.stat()

    return str(path), info.st_mtime_ns, info.st_size


def read_csv(source):
    raw = (
        source
        if isinstance(source, bytes)
        else Path(source).read_bytes()
    )

    for encoding in ("utf-8-sig", "gb18030", "utf-16"):
        try:
            text = raw.decode(encoding)

        except UnicodeError:
            continue

        header = [
            column.strip()
            for column in next(
                csv.reader(io.StringIO(text)),
                [],
            )
        ]

        if not header:
            raise ValueError("The CSV file is empty.")

        if len(header) != len(set(header)):
            raise ValueError("Duplicate CSV column names.")

        frame = pd.read_csv(
            io.StringIO(text),
            low_memory=False,
        )

        frame.columns = frame.columns.str.strip()

        return frame

    raise ValueError(
        "Please save the CSV with UTF-8 encoding."
    )


def numeric(frame):
    # Keep the same numeric conversion as the original platform.
    def convert(value):
        try:
            value = float(
                str(value).strip().replace("−", "-")
            )

            return value if np.isfinite(value) else np.nan

        except (TypeError, ValueError):
            return np.nan

    return frame.apply(
        lambda series: series.map(convert)
    )


def json_list(value):
    if isinstance(value, list):
        return value

    if pd.isna(value) or not str(value).strip():
        return []

    result = json.loads(value)

    if not isinstance(result, list):
        raise ValueError("Invalid library metadata.")

    return result


# ============================================================
# 7. SMILES and parent-structure matching
# ============================================================

@lru_cache(maxsize=50000)
def canonical(value):
    text = (
        str(value)
        .strip()
        .replace("\u200b", "")
        .replace("\ufeff", "")
    )

    if not text or any(char.isspace() for char in text):
        raise ValueError(
            "Invalid SMILES: remove internal spaces "
            "and trailing names."
        )

    parser = Chem.SmilesParserParams()
    parser.parseName = False
    parser.allowCXSMILES = False

    mol = Chem.MolFromSmiles(text, parser)

    if mol is None or mol.GetNumAtoms() == 0:
        raise ValueError(
            "Invalid SMILES. "
            "Please check the molecular structure."
        )

    for atom in mol.GetAtoms():
        atom.SetAtomMapNum(0)

    return Chem.MolToSmiles(
        mol,
        canonical=True,
        isomericSmiles=True,
    )


@lru_cache(maxsize=50000)
def parent_key(value):
    # Used only for identity matching.
    # The original selected structure remains the RDKit input.
    mol = MS.Cleanup(
        Chem.MolFromSmiles(canonical(value))
    )

    uncharger = MS.Uncharger()

    parts = [
        Chem.MolToSmiles(
            uncharger.uncharge(fragment),
            isomericSmiles=True,
        )
        for fragment in Chem.GetMolFrags(
            mol,
            asMols=True,
            sanitizeFrags=True,
        )
    ]

    if len(parts) == 1:
        return parts[0]

    organic = [
        part
        for part in parts
        if part not in INERT
    ]

    return (
        organic[0]
        if organic and len(set(organic)) == 1
        else ""
    )


def same_parent(left, right):
    if canonical(left) == canonical(right):
        return True

    key = parent_key(left)

    return bool(
        key and key == parent_key(right)
    )


# ============================================================
# 8. DFT descriptor validation
# ============================================================

def check_dft(values):
    values = np.asarray(values, dtype=float)

    if (
        values.shape != (13,)
        or not np.isfinite(values).all()
    ):
        raise ValueError(
            "All 13 DFT descriptors must be "
            "finite numeric values."
        )

    (
        homo,
        lumo,
        gap,
        ip,
        ea,
        chi,
        mu,
        eta,
        soft,
        omega,
    ) = values[:10]

    if gap <= 0 or eta <= 0 or soft <= 0 or omega < 0:
        raise ValueError(
            "Check the gap, hardness, softness "
            "and electrophilicity."
        )

    # Match this study's training definitions.
    checks = [
        ("Egap", gap, lumo - homo),
        ("chi", chi, (ip + ea) / 2),
        ("mu", mu, -chi),
        ("eta", eta, (ip - ea) / 2),
        ("S", soft, 1 / (2 * eta)),
        ("omega", omega, chi * chi / (2 * eta)),
    ]

    failed = [
        name
        for name, actual, expected in checks
        if not np.isclose(
            actual,
            expected,
            atol=5e-4,
            rtol=5e-4,
        )
    ]

    if failed:
        raise ValueError(
            "Check descriptor definitions/units: "
            + ", ".join(failed)
        )

    return dict(zip(DFT, values.tolist()))


# ============================================================
# 9. Load the self-contained DFT library
# ============================================================

@cached(
    "cache_data",
    show_spinner=False,
    max_entries=2,
)
def load_library(stamp):
    frame = read_csv(Path(stamp[0]))

    required = {
        "library_id",
        "canonical_smiles",
        "compound_name",
        "dft_input_smiles",
        "name_aliases_json",
        "cas_aliases_json",
        "source_datasets_json",
        "source_records_json",
        *DFT,
    }

    missing = sorted(
        required - set(frame.columns)
    )

    if missing:
        raise ValueError(
            "Missing library columns: "
            + ", ".join(missing)
        )

    accepted = []
    rejected = 0
    rejected_sources = 0

    for _, original in frame.iterrows():
        row = original.copy()

        try:
            smiles = canonical(
                row["canonical_smiles"]
            )

            check_dft(
                row[DFT].to_numpy(float)
            )

            if not same_parent(
                smiles,
                row["dft_input_smiles"],
            ):
                raise ValueError(
                    "DFT structure mismatch."
                )

            records = json_list(
                row["source_records_json"]
            )

            if not records:
                raise ValueError(
                    "Missing provenance."
                )

            valid = []

            for record in records:
                try:
                    if canonical(
                        record["source_smiles"]
                    ) != smiles:
                        raise ValueError(
                            "Source structure mismatch."
                        )

                    if not same_parent(
                        smiles,
                        record["dft_input_smiles"],
                    ):
                        raise ValueError(
                            "Source/DFT parent mismatch."
                        )

                    valid.append(record)

                except (
                    ValueError,
                    TypeError,
                    KeyError,
                    RuntimeError,
                ):
                    rejected_sources += 1

            if not valid:
                raise ValueError(
                    "No valid source mappings."
                )

            # Source paths are metadata only.
            # Do not open record["file"] on the server.
            if len(valid) != len(records):
                # Avoid aliases potentially introduced
                # by invalid source mappings.
                row["name_aliases_json"] = "[]"
                row["cas_aliases_json"] = "[]"
                row["compound_name"] = smiles

            row["source_records_json"] = json.dumps(valid)
            row["source_record_count"] = len(valid)

            row["source_datasets_json"] = json.dumps(
                sorted({
                    record["dataset"]
                    for record in valid
                })
            )

            json_list(row["name_aliases_json"])
            json_list(row["cas_aliases_json"])
            json_list(row["source_datasets_json"])

            row["canonical_smiles"] = smiles
            row["parent_key"] = parent_key(smiles)

            accepted.append(row)

        except (
            ValueError,
            TypeError,
            KeyError,
            RuntimeError,
        ):
            rejected += 1

    if not accepted:
        raise ValueError(
            "No valid DFT library records."
        )

    library = pd.DataFrame(
        accepted
    ).reset_index(drop=True)

    names = {}
    cases = {}

    for _, row in library.iterrows():
        aliases = (
            json_list(row["name_aliases_json"])
            + [row["compound_name"]]
        )

        for index, values in (
            (names, aliases),
            (
                cases,
                json_list(row["cas_aliases_json"]),
            ),
        ):
            for value in values:
                if (
                    pd.notna(value)
                    and str(value).strip()
                ):
                    index.setdefault(
                        norm(value),
                        set(),
                    ).add(
                        row["canonical_smiles"]
                    )

    if rejected:
        print(
            f"[Library] Excluded {rejected} "
            "invalid records.",
            flush=True,
        )

    if rejected_sources:
        print(
            f"[Library] Excluded {rejected_sources} "
            "invalid source mappings.",
            flush=True,
        )

    return library, names, cases


# ============================================================
# 10. Compound identifier resolution
# ============================================================

@cached(
    "cache_data",
    ttl=86400,
    show_spinner=False,
    max_entries=1000,
)
def pubchem_structures(identifier):
    url = (
        "https://pubchem.ncbi.nlm.nih.gov/rest/pug/"
        "compound/name/"
        + quote(identifier, safe="")
        + "/property/IsomericSMILES/JSON"
    )

    request = Request(
        url,
        headers={
            "User-Agent": "ChlorinePredictor/3.0"
        },
    )

    with urlopen(request, timeout=15) as response:
        payload = json.load(response)

    structures = set()

    items = payload.get(
        "PropertyTable",
        {},
    ).get(
        "Properties",
        [],
    )

    for item in items:
        for key in (
            "SMILES",
            "IsomericSMILES",
            "CanonicalSMILES",
            "ConnectivitySMILES",
        ):
            if item.get(key):
                structures.add(
                    canonical(item[key])
                )
                break

    if not structures:
        raise ValueError(
            "No structure found."
        )

    return sorted(structures)


def resolve(identifier, kind, names, cases):
    if not identifier.strip():
        raise ValueError(
            "Enter a compound name, "
            "CAS number or SMILES."
        )

    if kind == "SMILES":
        return [canonical(identifier)]

    index = (
        names
        if kind == "Chemical name"
        else cases
    )

    normalized = norm(identifier)

    if normalized in index:
        return sorted(index[normalized])

    try:
        return pubchem_structures(
            identifier.strip()
        )

    except Exception as exc:
        raise ValueError(
            "Compound not resolved. "
            "Please enter SMILES."
        ) from exc


# ============================================================
# 11. Library record selection
# ============================================================

def library_hits(library, smiles, system):
    exact = library[
        "canonical_smiles"
    ].eq(smiles)

    key = parent_key(smiles)

    related = (
        library["parent_key"].eq(key)
        if key
        else exact
    )

    hits = library.loc[
        exact | related
    ].copy()

    hits["_exact"] = hits[
        "canonical_smiles"
    ].eq(smiles)

    hits["_preferred"] = hits[
        "source_datasets_json"
    ].map(
        lambda value: (
            CONFIG[system]["tag"]
            in json_list(value)
        )
    )

    hits = hits.sort_values(
        [
            "_exact",
            "_preferred",
            "library_id",
        ],
        ascending=[
            False,
            False,
            True,
        ],
    )

    # Priority:
    # 1. Exact structure
    # 2. Selected-system source
    # 3. Stable library ID
    #
    # Distinct descriptor versions are not averaged.
    return hits


# ============================================================
# 12. DFT CSV upload
# ============================================================

def upload_rows(raw, smiles):
    if len(raw) > 5 * 1024 * 1024:
        raise ValueError(
            "Upload a CSV smaller than 5 MB."
        )

    frame = read_csv(raw)

    if not 1 <= len(frame) <= 10000:
        raise ValueError(
            "The CSV must contain "
            "1–10000 data rows."
        )

    for old, new in zip(ASCII_DFT, DFT):
        if old in frame:
            if new in frame:
                raise ValueError(
                    "Keep only one column for: "
                    + new
                )

            frame = frame.rename(
                columns={old: new}
            )

    missing = sorted(
        set(DFT) - set(frame.columns)
    )

    if missing:
        raise ValueError(
            "Missing descriptors: "
            + ", ".join(missing)
        )

    identities = [
        column
        for column in (
            "smiles",
            "SMILES",
            "canonical_smiles",
            "dft_input_smiles",
            "DFT用SMILES",
        )
        if column in frame
    ]

    if identities:
        def matches(row):
            values = [
                row[column]
                for column in identities
                if (
                    pd.notna(row[column])
                    and str(row[column]).strip()
                )
            ]

            try:
                return (
                    bool(values)
                    and all(
                        same_parent(value, smiles)
                        for value in values
                    )
                )

            except Exception:
                return False

        frame = frame.loc[
            frame.apply(
                matches,
                axis=1,
            )
        ].copy()

        if frame.empty:
            raise ValueError(
                "No matching compound/parent "
                "found in the CSV."
            )

    elif len(frame) != 1:
        raise ValueError(
            "A multirow CSV must include "
            "a SMILES column."
        )

    return frame


def uploaded_dft(row):
    if "energy_unit" in row:
        if norm(row["energy_unit"]) != "ev":
            raise ValueError(
                "Energy descriptors must use eV."
            )

    if "softness_unit" in row:
        unit = norm(
            row["softness_unit"]
        ).replace(" ", "")

        if unit not in {
            "ev^-1",
            "ev-1",
            "1/ev",
        }:
            raise ValueError(
                "Softness must use eV^-1."
            )

    values = numeric(
        pd.DataFrame([
            row[DFT].to_dict()
        ])
    )[DFT].iloc[0]

    return check_dft(values)


def template_bytes(smiles):
    row = {
        "smiles": smiles,
        **dict.fromkeys(ASCII_DFT, ""),
    }

    row.update(
        energy_unit="eV",
        softness_unit="eV^-1",
    )

    return pd.DataFrame(
        [row]
    ).to_csv(
        index=False
    ).encode(
        "utf-8-sig"
    )


# ============================================================
# 13. Model fitting and prediction
# ============================================================

def make_model(system):
    config = CONFIG[system]

    threads = max(
        1,
        min(
            4,
            os.cpu_count() or 1,
        ),
    )

    if config["algorithm"] == "XGB":
        from xgboost import XGBRegressor

        return XGBRegressor(
            objective="reg:squarederror",
            random_state=42,
            n_jobs=threads,
            tree_method="hist",
            verbosity=0,
            **config["params"],
        )

    from catboost import CatBoostRegressor

    return CatBoostRegressor(
        loss_function="RMSE",
        eval_metric="R2",
        bootstrap_type="Bayesian",
        random_seed=42,
        verbose=False,
        allow_writing_files=False,
        thread_count=threads,
        **config["params"],
    )


@cached(
    "cache_resource",
    show_spinner=False,
    max_entries=6,
)
def model_bundle(
    system,
    with_dft,
    stamp,
    config_version,
):
    features = (
        STRUCT
        + (DFT if with_dft else [])
        + ["pH", "T"]
    )

    raw = read_csv(
        Path(stamp[0])
    )

    missing = sorted(
        set(features + ["logk"])
        - set(raw.columns)
    )

    if missing:
        raise ValueError(
            "Missing training columns: "
            + ", ".join(missing)
        )

    data = numeric(
        raw[features + ["logk"]]
    )

    data = data.loc[
        data["logk"].notna()
    ]

    if (
        len(data) < 2
        or data[features].isna().all().any()
    ):
        raise ValueError(
            "Invalid reference dataset "
            "or entirely missing feature."
        )

    imputer = SimpleImputer(
        strategy="median"
    )

    X = pd.DataFrame(
        imputer.fit_transform(
            data[features]
        ),
        columns=features,
    )

    model = make_model(system)

    # Deployment fitting on all available reference records.
    # This does not replace the manuscript's held-out evaluation.
    model.fit(
        X,
        data["logk"],
    )

    return (
        model,
        imputer,
        features,
        threading.Lock(),
    )


def predict(system, smiles, ph, temp, dft):
    mol = Chem.MolFromSmiles(
        canonical(smiles)
    )

    functions = dict(
        Descriptors.descList
    )

    values = {
        name: float(functions[name](mol))
        for name in STRUCT
    }

    values.update(dft or {})

    values.update(
        pH=float(ph),
        T=float(temp),
    )

    path = DATA / CONFIG[system]["file"]

    model, imputer, features, lock = model_bundle(
        system,
        dft is not None,
        file_stamp(path),
        digest(CONFIG[system]),
    )

    X = pd.DataFrame(
        [[values[name] for name in features]],
        columns=features,
    )

    if not np.isfinite(
        X.to_numpy(float)
    ).all():
        raise ValueError(
            "The structure produced "
            "invalid descriptors."
        )

    X = pd.DataFrame(
        imputer.transform(X),
        columns=features,
    )

    with lock:
        result = float(
            model.predict(X)[0]
        )

    if not np.isfinite(result):
        raise ValueError(
            "The model returned "
            "an invalid prediction."
        )

    return result


def format_k(logk):
    if -2 <= logk < 6:
        return f"{10.0 ** logk:,.2f}"

    exponent = math.floor(logk)
    mantissa = 10.0 ** (logk - exponent)

    if round(mantissa, 2) >= 10:
        mantissa /= 10
        exponent += 1

    return (
        f"{mantissa:.2f} × 10^{exponent}"
    )


# ============================================================
# 14. Session-state utilities
# ============================================================

def clear_result():
    st.session_state.pop(
        "compact_result",
        None,
    )


def clear_compound():
    st.session_state.pop(
        "compact_prepared",
        None,
    )

    clear_result()


# ============================================================
# 15. Main interface
# ============================================================

def run():
    st.markdown(
        CSS,
        unsafe_allow_html=True,
    )

    st.markdown(
        '<div class="app-banner">'
        "Active Chlorine Reactivity Prediction Platform"
        "</div>",
        unsafe_allow_html=True,
    )

    required = (
        [LIBRARY]
        + [
            DATA / config["file"]
            for config in CONFIG.values()
        ]
    )

    missing = [
        path.name
        for path in required
        if not path.is_file()
    ]

    if missing:
        st.error(
            "Upload the following files to data/:"
        )

        st.code(
            "\n".join(missing),
            language="text",
        )

        return

    try:
        with st.spinner(
            "Loading descriptor library..."
        ):
            library, names, cases = load_library(
                file_stamp(LIBRARY)
            )

    except Exception as exc:
        st.error(
            "Cannot load the DFT library: "
            + str(exc)
        )

        return

    left, right = st.columns(
        2,
        gap="large",
    )

    with left:
        input_panel = st.container(
            border=True
        )

        analysis_panel = st.container(
            border=True
        )

    with right:
        compound_panel = st.container(
            border=True
        )

        prediction_panel = st.container(
            border=True
        )

    # --------------------------------------------------------
    # Module 1. Input
    # --------------------------------------------------------

    with input_panel:
        st.subheader("1. Input")

        system = st.selectbox(
            "Active chlorine system",
            list(CONFIG),
            key="input_system",
            on_change=clear_result,
        )

        kind = st.radio(
            "Input type",
            ["Chemical name", "CAS", "SMILES"],
            horizontal=True,
            key="input_kind",
            on_change=clear_compound,
        )

        identifier = st.text_input(
            "Compound",
            placeholder="e.g., Bisphenol A",
            key="input_compound",
            on_change=clear_compound,
        )

        col_a, col_b = st.columns(2)

        ph = col_a.number_input(
            "pH",
            min_value=0.0,
            max_value=14.0,
            value=7.0,
            step=0.1,
            key="input_ph",
            on_change=clear_result,
        )

        temp = col_b.number_input(
            "Temperature (°C)",
            min_value=-50.0,
            max_value=150.0,
            value=25.0,
            step=1.0,
            key="input_temp",
            on_change=clear_result,
        )

        if st.button(
            "Check compound",
            type="primary",
            use_container_width=True,
        ):
            clear_compound()

            try:
                with st.spinner(
                    "Finding structure..."
                ):
                    structures = resolve(
                        identifier,
                        kind,
                        names,
                        cases,
                    )

                serial = (
                    st.session_state.get(
                        "resolve_serial",
                        0,
                    )
                    + 1
                )

                st.session_state[
                    "resolve_serial"
                ] = serial

                st.session_state[
                    "compact_prepared"
                ] = (
                    kind,
                    identifier.strip(),
                    structures,
                    serial,
                )

            except Exception as exc:
                st.error(str(exc))

    prepared = st.session_state.get(
        "compact_prepared"
    )

    ready = bool(
        prepared
        and prepared[:2] == (
            kind,
            identifier.strip(),
        )
    )

    smiles = None
    confirmed = False
    context = ""

    # --------------------------------------------------------
    # Module 2. Confirm compound
    # --------------------------------------------------------

    with compound_panel:
        st.subheader(
            "2. Confirm compound"
        )

        if ready:
            structures = prepared[2]

            context = digest((
                kind,
                identifier.strip(),
                prepared[3],
            ))

            if len(structures) > 1:
                smiles = st.selectbox(
                    "Molecular structure",
                    structures,
                    format_func=lambda value: (
                        f"Structure "
                        f"{structures.index(value) + 1}"
                    ),
                    key="structure_" + context,
                    on_change=clear_result,
                )

            else:
                smiles = structures[0]

            exact = library.loc[
                library[
                    "canonical_smiles"
                ].eq(smiles)
            ]

            if len(exact):
                name = str(
                    exact.iloc[0][
                        "compound_name"
                    ]
                )

            elif kind != "SMILES":
                name = identifier.strip()

            else:
                name = "Selected compound"

            st.markdown(
                '<div class="compound-name">'
                + html.escape(name)
                + "</div>",
                unsafe_allow_html=True,
            )

            st.image(
                Draw.MolToImage(
                    Chem.MolFromSmiles(smiles),
                    size=(440, 160),
                ),
                use_column_width="auto",
            )

            confirmed = st.checkbox(
                "Confirm this compound",
                key=(
                    "confirm_"
                    + digest((context, smiles))
                ),
                on_change=clear_result,
            )

        else:
            st.markdown(
                '<div class="empty-panel">'
                "Molecular structure"
                "</div>",
                unsafe_allow_html=True,
            )

    # --------------------------------------------------------
    # Module 3. Analysis mode
    # --------------------------------------------------------

    dft = None
    route = ROUTES[0]
    record_id = ""
    analysis_ok = False

    with analysis_panel:
        st.subheader(
            "3. Analysis mode"
        )

        if ready:
            analysis_context = digest((
                context,
                system,
                smiles,
            ))

            hits = library_hits(
                library,
                smiles,
                system,
            )

            route = st.radio(
                "Prediction route",
                ROUTES,
                index=0 if len(hits) else 2,
                key="route_" + analysis_context,
                label_visibility="collapsed",
                on_change=clear_result,
            )

            try:
                if route == ROUTES[0]:
                    if hits.empty:
                        st.warning(
                            "Library DFT data "
                            "not available."
                        )

                    else:
                        row = hits.iloc[0]

                        record_id = str(
                            row["library_id"]
                        )

                        dft = check_dft(
                            row[DFT].to_numpy(float)
                        )

                        analysis_ok = True

                        st.success(
                            "Library DFT data ready."
                        )

                elif route == ROUTES[1]:
                    uploaded = st.file_uploader(
                        "DFT descriptors",
                        type=["csv"],
                        key="upload_" + analysis_context,
                        on_change=clear_result,
                    )

                    st.download_button(
                        "CSV template",
                        template_bytes(smiles),
                        file_name="DFT_template.csv",
                        mime="text/csv",
                    )

                    if uploaded is not None:
                        raw = uploaded.getvalue()

                        rows = upload_rows(
                            raw,
                            smiles,
                        )

                        record_key = digest(raw)

                        if len(rows) > 1:
                            index = st.selectbox(
                                "CSV record",
                                list(rows.index),
                                index=None,
                                placeholder="Select a row",
                                format_func=lambda value: (
                                    f"Data row {value + 2}"
                                ),
                                key=(
                                    "csv_row_"
                                    + analysis_context
                                    + record_key
                                ),
                                on_change=clear_result,
                            )

                        else:
                            index = rows.index[0]

                        if index is not None:
                            dft = uploaded_dft(
                                rows.loc[index]
                            )

                            record_id = digest((
                                record_key,
                                int(index),
                            ))

                            analysis_ok = st.checkbox(
                                "Confirm DFT data for this compound",
                                key=(
                                    "csv_confirm_"
                                    + analysis_context
                                    + record_id
                                ),
                                on_change=clear_result,
                                help=(
                                    "Use the study's DFT protocol; "
                                    "energy: eV; softness: eV^-1."
                                ),
                            )

                else:
                    analysis_ok = True

                    st.caption(
                        "Screening prediction "
                        "without DFT descriptors."
                    )

            except Exception as exc:
                analysis_ok = False
                st.error(str(exc))

        else:
            st.radio(
                "Prediction route",
                ROUTES,
                disabled=True,
                label_visibility="collapsed",
                key="route_pending",
            )

    # --------------------------------------------------------
    # Module 4. Prediction
    # --------------------------------------------------------

    with prediction_panel:
        st.subheader(
            "4. Prediction"
        )

        token = digest((
            context,
            system,
            smiles,
            float(ph),
            float(temp),
            route,
            record_id,
            dft,
        ))

        enabled = (
            ready
            and confirmed
            and analysis_ok
        )

        if st.button(
            "Predict",
            type="primary",
            use_container_width=True,
            disabled=not enabled,
        ):
            clear_result()

            try:
                with st.spinner(
                    "Preparing model and predicting..."
                ):
                    value = predict(
                        system,
                        smiles,
                        ph,
                        temp,
                        dft,
                    )

                st.session_state[
                    "compact_result"
                ] = (
                    token,
                    value,
                )

            except Exception as exc:
                st.error(
                    "Prediction failed: "
                    + str(exc)
                )

        result = st.session_state.get(
            "compact_result"
        )

        value = (
            result[1]
            if (
                enabled
                and result
                and result[0] == token
            )
            else None
        )

        st.metric(
            "log k",
            (
                f"{value:.2f}"
                if value is not None
                else "—"
            ),
        )

        st.metric(
            "k (M⁻¹ s⁻¹)",
            (
                format_k(value)
                if value is not None
                else "—"
            ),
        )


# ============================================================
# 16. Spyder launcher
# ============================================================

def launch_local():
    app_file = Path(
        __file__
    ).resolve()

    previous = globals().get(
        "_CHLORINE_SERVER"
    )

    if (
        previous
        and previous[0] == str(app_file)
        and previous[1].poll() is None
    ):
        print(
            "Platform:",
            previous[2],
            flush=True,
        )

        webbrowser.open(previous[2])
        return

    executable = Path(
        sys.executable
    )

    if executable.name.lower() == "pythonw.exe":
        executable = executable.with_name(
            "python.exe"
        )

    port = None

    for candidate in range(8501, 8521):
        with socket.socket() as sock:
            try:
                sock.bind((
                    "127.0.0.1",
                    candidate,
                ))

                port = candidate
                break

            except OSError:
                continue

    if port is None:
        raise RuntimeError(
            "No free port between 8501 and 8520."
        )

    url = f"http://127.0.0.1:{port}"

    command = [
        str(executable),
        "-m",
        "streamlit",
        "run",
        str(app_file),

        "--server.port",
        str(port),

        "--server.address",
        "127.0.0.1",

        "--server.headless",
        "true",

        "--server.runOnSave",
        "true",

        "--browser.gatherUsageStats",
        "false",

        "--runner.magicEnabled",
        "false",
    ]

    process = subprocess.Popen(
        command,
        cwd=str(app_file.parent),
        stdout=subprocess.PIPE,
        stderr=subprocess.STDOUT,
        text=True,
        encoding="utf-8",
        errors="replace",
        env={
            **os.environ,
            "PYTHONIOENCODING": "utf-8",
        },
        creationflags=(
            subprocess.CREATE_NO_WINDOW
            if os.name == "nt"
            else 0
        ),
    )

    globals()["_CHLORINE_SERVER"] = (
        str(app_file),
        process,
        url,
    )

    print(
        "Starting platform:",
        url,
        flush=True,
    )

    def logs():
        for line in process.stdout:
            print(
                "[Streamlit] "
                + line.rstrip(),
                flush=True,
            )

    def open_when_ready():
        for _ in range(120):
            if process.poll() is not None:
                print(
                    "Startup failed. Check the log.",
                    flush=True,
                )

                return

            try:
                with urlopen(
                    url + "/_stcore/health",
                    timeout=1,
                ) as response:
                    if response.status == 200:
                        print(
                            "Platform ready:",
                            url,
                            flush=True,
                        )

                        webbrowser.open(url)
                        return

            except Exception:
                pass

            time.sleep(0.5)

        print(
            "Startup is taking longer than expected. "
            "Check the log.",
            flush=True,
        )

    threading.Thread(
        target=logs,
        daemon=True,
    ).start()

    threading.Thread(
        target=open_when_ready,
        daemon=True,
    ).start()


# ============================================================
# 17. Entry point
# ============================================================

if __name__ == "__main__":
    if IN_STREAMLIT:
        run()
    else:
        launch_local()