"""
Goodreads Sanity Checks
=======================
Validation tests for the Goodreads recommendation pipeline outputs.

Run using: python -m pytest -q 03d_sanity_checks.py

These tests verify:
- Sampled preferences are valid full permutations
- Consensus rankings are properly formatted
- No accidental file duplication
- Methods produce different outputs
"""

import os
import glob
import hashlib
import pickle
import pandas as pd
import pytest
import numpy as np 

from utils.io import load_consensus_ranking, load_sampled_preferences 


# =============================================================================
# Helpers
# =============================================================================

def sha256_file(path: str) -> str:
    h = hashlib.sha256()
    with open(path, "rb") as f:
        for chunk in iter(lambda: f.read(1024 * 1024), b""):
            h.update(chunk)
    return h.hexdigest()


def canonical_committee_signature(committee) -> str:
    """
    Stable signature for a parsed committee list.
    If two committees have same signature, they are identical sequences.
    """
    data = ",".join(map(str, committee)).encode("utf-8")
    return hashlib.sha256(data).hexdigest()


def load_all_committees(consensus_dir: str):
    paths = sorted(glob.glob(os.path.join(consensus_dir, "*.txt")))
    assert paths, f"No .txt consensus files found in: {consensus_dir}"
    committees = {}
    for p in paths:
        name = os.path.splitext(os.path.basename(p))[0]
        committees[name] = load_consensus_ranking(p)
    return paths, committees


# =============================================================================
# Fixtures (configure via env vars)
# =============================================================================

@pytest.fixture(scope="session")
def consensus_dir():
    """
    Point this at one sample directory that contains *.txt consensus outputs.
    Set via env var CONSENSUS_DIR or use default.
    """
    return os.environ.get(
        "CONSENSUS_DIR",
        "consensus_results/sample_0",
    )


@pytest.fixture(scope="session")
def pref_path():
    """
    Set PREF_PATH env var to control which file gets tested.
    """
    return os.environ.get(
        "PREF_PATH",
        "consensus_results/sample_0/sampled_rankings.pkl"
    )


@pytest.fixture(scope="session")
def sampled_rankings_paths():
    """
    Finds multiple sampled_rankings.pkl files.
    Control via env vars: BASE_SAMPLES_DIR, SAMPLE_GLOB, MAX_SAMPLES
    """
    base = os.environ.get(
        "BASE_SAMPLES_DIR",
        "consensus_results"
    )
    glob_pat = os.environ.get("SAMPLE_GLOB", "sample_*/sampled_rankings.pkl")
    max_samples = int(os.environ.get("MAX_SAMPLES", "20"))

    paths = sorted(glob.glob(os.path.join(base, glob_pat)))
    if len(paths) < 2:
        pytest.skip(f"Need >=2 sampled_rankings.pkl files, found {len(paths)}")

    return paths[:max_samples]


# =============================================================================
# Test Helpers
# =============================================================================

def _infer_universe_from_preferences(preferences: pd.DataFrame):
    """
    Universe inferred from the union of all items appearing in any user's ranking.
    Returns a sorted list of unique item ids.
    """
    exploded = preferences.explode("Ranked_Items")
    items = exploded["Ranked_Items"].dropna().tolist()
    assert all(isinstance(x, (int, np.integer)) for x in items), (
        "Non-integer item id found in rankings."
    )
    return sorted(set(int(x) for x in items))


def _check_rankings_are_full_permutations(preferences: pd.DataFrame, universe=None):
    """
    Assert that each user's Ranked_Items list is a permutation of `universe`.
    """
    assert "User_ID" in preferences.columns
    assert "Ranked_Items" in preferences.columns

    user_lists = preferences.groupby("User_ID")["Ranked_Items"].first()

    if universe is None:
        universe = _infer_universe_from_preferences(preferences)

    Uset = set(universe)
    d = len(universe)
    assert d > 0

    bad = []
    for uid, r in user_lists.items():
        assert isinstance(r, (list, tuple)), f"User {uid} Ranked_Items is not list/tuple"

        if not all(isinstance(x, (int, np.integer)) for x in r):
            bad.append((uid, "non-int item(s)", r))
            continue

        r = [int(x) for x in r]

        if len(r) != d:
            missing = sorted(Uset - set(r))
            extra = sorted(set(r) - Uset)
            bad.append((uid, f"len={len(r)} expected={d}", {"missing": missing[:20], "extra": extra[:20]}))
            continue

        if len(set(r)) != len(r):
            seen = set()
            dupes = []
            for x in r:
                if x in seen:
                    dupes.append(x)
                seen.add(x)
            bad.append((uid, "duplicates", dupes[:20]))
            continue

        if set(r) != Uset:
            missing = sorted(Uset - set(r))
            extra = sorted(set(r) - Uset)
            bad.append((uid, "not equal to universe", {"missing": missing[:20], "extra": extra[:20]}))

    assert not bad, (
        "Some user rankings are not full permutations of the expected universe.\n"
        + "\n".join([f"User {uid}: {reason} -> {info}" for uid, reason, info in bad[:25]])
    )


def _df_fingerprint_preferences(df: pd.DataFrame) -> str:
    """
    Create a stable fingerprint for preferences.
    """
    if not isinstance(df, pd.DataFrame):
        raise TypeError("preferences must be a DataFrame")

    if "User_ID" not in df.columns or "Ranked_Items" not in df.columns:
        raise ValueError("preferences must contain User_ID and Ranked_Items")

    canon = (
        df[["User_ID", "Ranked_Items"]]
        .copy()
        .sort_values("User_ID", kind="mergesort")
        .reset_index(drop=True)
    )
    canon["Ranked_Items"] = canon["Ranked_Items"].apply(lambda x: tuple(x))

    payload = canon.to_csv(index=False).encode("utf-8")
    return hashlib.sha256(payload).hexdigest()


# =============================================================================
# Tests
# =============================================================================

def test_rankings_all_same_length_and_permutations(pref_path):
    """
    Single-file check: every user ranking is same length, no duplicates.
    """
    if not os.path.exists(pref_path):
        pytest.skip(f"Preferences file not found: {pref_path}")
    preferences = load_sampled_preferences(pref_path)
    _check_rankings_are_full_permutations(preferences, universe=None)


def test_preferences_have_multiple_users_and_items(pref_path):
    """
    Verify we have sufficient users and items.
    """
    if not os.path.exists(pref_path):
        pytest.skip(f"Preferences file not found: {pref_path}")
    preferences = load_sampled_preferences(pref_path)

    assert isinstance(preferences, pd.DataFrame)
    assert "User_ID" in preferences.columns
    assert "Ranked_Items" in preferences.columns

    n_users = preferences["User_ID"].nunique()
    assert n_users > 1, f"Only {n_users} unique user(s) found"

    exploded = preferences.explode("Ranked_Items")
    n_items = exploded["Ranked_Items"].nunique()
    assert n_items > 1, f"Only {n_items} unique item(s) found"

    user_lists = preferences.groupby("User_ID")["Ranked_Items"].first()
    n_unique_rank_lists = user_lists.apply(tuple).nunique()
    assert n_unique_rank_lists > 1, "All users have identical rankings"


def test_multiple_sampled_rankings_differ(sampled_rankings_paths):
    """
    Sanity check: sampled_rankings across samples should not all be identical.
    """
    fps = []
    for p in sampled_rankings_paths:
        prefs = load_sampled_preferences(p)
        fps.append((_df_fingerprint_preferences(prefs), p))

    seen = {}
    collisions = []
    for fp, path in fps:
        if fp in seen:
            collisions.append((seen[fp], path))
        else:
            seen[fp] = path

    assert not collisions, (
        "Found identical sampled_rankings across different samples.\n"
        + "\n".join([f"{a} == {b}" for a, b in collisions])
    )


def test_consensus_files_exist(consensus_dir):
    """
    Verify consensus files are present.
    """
    if not os.path.exists(consensus_dir):
        pytest.skip(f"Consensus directory not found: {consensus_dir}")
    paths = sorted(glob.glob(os.path.join(consensus_dir, "*.txt")))
    assert len(paths) > 0, f"No .txt files found in {consensus_dir}"


def test_load_consensus_ranking_parses_nonempty(consensus_dir):
    """
    Verify consensus files can be parsed.
    """
    if not os.path.exists(consensus_dir):
        pytest.skip(f"Consensus directory not found: {consensus_dir}")
    paths = sorted(glob.glob(os.path.join(consensus_dir, "*.txt")))
    if not paths:
        pytest.skip("No consensus files to test")
    committee = load_consensus_ranking(paths[0])
    assert isinstance(committee, list)
    assert len(committee) > 0, f"Parsed empty committee from {paths[0]}"


def test_committees_are_ints_and_unique(consensus_dir):
    """
    Verify committee entries are integers with no duplicates.
    """
    if not os.path.exists(consensus_dir):
        pytest.skip(f"Consensus directory not found: {consensus_dir}")
    paths = sorted(glob.glob(os.path.join(consensus_dir, "*.txt")))
    if not paths:
        pytest.skip("No consensus files to test")
    
    _, committees = load_all_committees(consensus_dir)

    for name, committee in committees.items():
        assert isinstance(committee, list), f"{name} committee is not a list"
        assert len(committee) > 0, f"{name} committee empty"
        assert all(isinstance(x, int) for x in committee), f"{name} has non-int entries"
        assert len(set(committee)) == len(committee), f"{name} committee contains duplicates"


def test_consensus_text_files_not_byte_identical(consensus_dir):
    """
    Check that files are not identical at the byte level.
    """
    if not os.path.exists(consensus_dir):
        pytest.skip(f"Consensus directory not found: {consensus_dir}")
    paths = sorted(glob.glob(os.path.join(consensus_dir, "*.txt")))
    if len(paths) < 2:
        pytest.skip("Need at least 2 consensus files to compare")
    
    hashes = {p: sha256_file(p) for p in paths}

    unique_hashes = set(hashes.values())
    assert len(unique_hashes) == len(paths), (
        "Some consensus .txt files are byte-identical.\n"
        + "\n".join(
            f"{h}: {[os.path.basename(p) for p,v in hashes.items() if v==h]}"
            for h in unique_hashes
            if sum(1 for v in hashes.values() if v == h) > 1
        )
    )


def test_most_committees_differ_at_topk(consensus_dir):
    """
    Verify methods differ in their first K items.
    """
    if not os.path.exists(consensus_dir):
        pytest.skip(f"Consensus directory not found: {consensus_dir}")
    TOPK = int(os.environ.get("TOPK", "20"))
    
    paths = sorted(glob.glob(os.path.join(consensus_dir, "*.txt")))
    if not paths:
        pytest.skip("No consensus files to test")
    
    _, committees = load_all_committees(consensus_dir)

    topk_sigs = set()
    for method, committee in committees.items():
        topk = tuple(committee[:TOPK])
        topk_sigs.add(topk)

    assert len(topk_sigs) >= 2, (
        f"All methods share the same top-{TOPK} ordering"
    )


def test_consensus_methods_cover_expected_count(consensus_dir):
    """
    Verify we have a minimum number of methods.
    """
    if not os.path.exists(consensus_dir):
        pytest.skip(f"Consensus directory not found: {consensus_dir}")
    MIN_METHODS = int(os.environ.get("MIN_METHODS", "5"))
    paths = sorted(glob.glob(os.path.join(consensus_dir, "*.txt")))
    assert len(paths) >= MIN_METHODS, (
        f"Expected at least {MIN_METHODS} consensus files, found {len(paths)}."
    )

