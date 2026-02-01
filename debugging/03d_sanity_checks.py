import os
import glob
import hashlib
import pickle
import pandas as pd
import pytest
import pickle 
import numpy as np 

from utils.io import load_consensus_ranking, load_sampled_preferences 

# RUN USING: python -m pytest -q 03d_sanity_checks.py 
# ---- helpers ----
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
    # include order; join as bytes
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

# ---- fixtures (configure these paths) ----
@pytest.fixture(scope="session")
def consensus_dir():
    """
    Point this at one sample directory that contains *.txt consensus outputs.
    You can also set it via env var to avoid editing.
    """
    return os.environ.get(
        "CONSENSUS_DIR",
        "/data2/rsalgani/Prefix/ml-1m/agg_files/sample_0",  # <-- CHANGE ME
    )

@pytest.fixture(scope="session")
def pref_path():
    """
    Set PREF_PATH env var to control which file gets tested, e.g.
      export PREF_PATH=/data2/.../sample_0/sampled_rankings.pkl
    """
    return os.environ.get(
        "PREF_PATH",
        "/data2/rsalgani/Prefix/ml-1m/agg_files/sample_0/sampled_rankings.pkl"  # <-- CHANGE ME
    )

@pytest.fixture(scope="session")
def sampled_rankings_paths():
    """
    Finds multiple sampled_rankings.pkl files.

    You can control via env vars:
      BASE_SAMPLES_DIR=/data2/.../agg_files
      SAMPLE_GLOB=sample_*/sampled_rankings.pkl
      MAX_SAMPLES=20
    """
    base = os.environ.get(
        "BASE_SAMPLES_DIR",
        "/data2/rsalgani/Prefix/ml-1m/agg_files"  # <-- CHANGE ME if needed
    )
    glob_pat = os.environ.get("SAMPLE_GLOB", "sample_*/sampled_rankings.pkl")
    max_samples = int(os.environ.get("MAX_SAMPLES", "20"))

    paths = sorted(glob.glob(os.path.join(base, glob_pat)))
    assert len(paths) >= 2, f"Need >=2 sampled_rankings.pkl files, found {len(paths)} at {base}/{glob_pat}"

    return paths[:max_samples]

# =============================================================================
# Tests
# =============================================================================

def _infer_universe_from_preferences(preferences: pd.DataFrame):
    """
    Universe inferred from the union of all items appearing in any user's ranking.
    Returns a sorted list of unique item ids.
    """
    exploded = preferences.explode("Ranked_Items")
    items = exploded["Ranked_Items"].dropna().tolist()
    # Be strict: require ints
    assert all(isinstance(x, (int, np.integer)) for x in items), (
        "Non-integer item id found in rankings. "
        "If your IDs are strings, cast them before writing sampled_rankings.pkl."
    )
    return sorted(set(int(x) for x in items))

def _check_rankings_are_full_permutations(preferences: pd.DataFrame, universe=None):
    """
    Assert that each user's Ranked_Items list is a permutation of `universe`.
    If universe is None, infer it from all users (union).
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
        # Must be list-like
        assert isinstance(r, (list, tuple)), f"User {uid} Ranked_Items is not list/tuple: {type(r)}"

        # All ints
        if not all(isinstance(x, (int, np.integer)) for x in r):
            bad.append((uid, "non-int item(s)", r))
            continue

        r = [int(x) for x in r]

        # Length must match universe size
        if len(r) != d:
            missing = sorted(Uset - set(r))
            extra = sorted(set(r) - Uset)
            bad.append((uid, f"len={len(r)} expected={d}", {"missing": missing[:20], "extra": extra[:20]}))
            continue

        # No duplicates
        if len(set(r)) != len(r):
            # find first few duplicates
            seen = set()
            dupes = []
            for x in r:
                if x in seen:
                    dupes.append(x)
                seen.add(x)
            bad.append((uid, "duplicates", dupes[:20]))
            continue

        # Must match universe exactly
        if set(r) != Uset:
            missing = sorted(Uset - set(r))
            extra = sorted(set(r) - Uset)
            bad.append((uid, "not equal to universe", {"missing": missing[:20], "extra": extra[:20]}))

    assert not bad, (
        "Some user rankings are not full permutations of the expected universe.\n"
        + "\n".join([f"User {uid}: {reason} -> {info}" for uid, reason, info in bad[:25]])
        + ("" if len(bad) <= 25 else f"\n...and {len(bad)-25} more.")
    )

def test_rankings_all_same_length_and_permutations(pref_path):
    """
    Single-file check: every user ranking is same length, no duplicates, and matches inferred universe.
    """
    preferences = load_sampled_preferences(pref_path)
    _check_rankings_are_full_permutations(preferences, universe=None)

# def test_rankings_contiguous_0_to_d_minus_1(pref_path):
#     """
#     Stronger check: item IDs must be exactly {0,1,...,d-1} (contiguous).
#     This matches the assumption in your ILP/tournament code.
#     """
#     preferences = load_sampled_preferences(pref_path)
#     universe = _infer_universe_from_preferences(preferences)
#     d = len(universe)

#     expected = list(range(d))
#     assert universe == expected, (
#         "Item IDs are not contiguous 0..d-1.\n"
#         f"Expected: {expected[:50]}{'...' if d>50 else ''}\n"
#         f"Found:    {universe[:50]}{'...' if d>50 else ''}\n"
#         "Fix by relabeling items to 0..d-1 during preprocessing."
#     )

# def test_all_sampled_rankings_have_valid_full_permutations(sampled_rankings_paths):
#     """
#     Batch check across multiple sample files.
#     Ensures all samples satisfy the permutation + equal-length requirement.
#     """
#     failures = []
#     for p in sampled_rankings_paths:
#         prefs = load_sampled_preferences(p)
#         try:
#             _check_rankings_are_full_permutations(prefs, universe=None)
#         except AssertionError as e:
#             failures.append((p, str(e)))

#     assert not failures, (
#         "Some sampled_rankings.pkl files contain invalid rankings.\n"
#         + "\n\n".join([f"{path}:\n{msg}" for path, msg in failures[:10]])
#         + ("" if len(failures) <= 10 else f"\n\n...and {len(failures)-10} more.")
#     )

# def test_all_sampled_rankings_contiguous_ids(sampled_rankings_paths):
#     """
#     Batch check: every sample uses contiguous IDs 0..d-1.
#     """
#     failures = []
#     for p in sampled_rankings_paths:
#         prefs = load_sampled_preferences(p)
#         universe = _infer_universe_from_preferences(prefs)
#         d = len(universe)
#         if universe != list(range(d)):
#             failures.append((p, universe[:50], d))

#     assert not failures, (
#         "Some samples have non-contiguous item IDs.\n"
#         + "\n".join([f"{path}: d={d}, first_ids={u}" for path, u, d in failures[:10]])
#         + ("" if len(failures) <= 10 else f"\n...and {len(failures)-10} more.")
#     )


def test_preferences_have_multiple_users_and_items(pref_path):
    preferences = load_sampled_preferences(pref_path)

    assert isinstance(preferences, pd.DataFrame), "preferences should be a pandas DataFrame"
    assert "User_ID" in preferences.columns, "preferences missing column: User_ID"
    assert "Ranked_Items" in preferences.columns, "preferences missing column: Ranked_Items"

    # Users
    n_users = preferences["User_ID"].nunique()
    assert n_users > 1, f"Only {n_users} unique user(s) found. Suspicious."

    # Items
    exploded = preferences.explode("Ranked_Items")
    n_items = exploded["Ranked_Items"].nunique()
    assert n_items > 1, f"Only {n_items} unique item(s) found. Suspicious."

    # Bonus: make sure not every user has identical ranking (optional but useful)
    # This catches accidental bugs where all users got copied
    user_lists = preferences.groupby("User_ID")["Ranked_Items"].first()
    n_unique_rank_lists = user_lists.apply(tuple).nunique()
    assert n_unique_rank_lists > 1, (
        "All users appear to have identical Ranked_Items lists. "
        "This might be expected in some synthetic tests, but is suspicious for real data."
    )

def _df_fingerprint_preferences(df: pd.DataFrame) -> str:
    """
    Create a stable fingerprint for preferences.
    We canonicalize to (User_ID, tuple(Ranked_Items)) sorted by User_ID.
    This catches accidental duplication across samples.
    """
    if not isinstance(df, pd.DataFrame):
        raise TypeError("preferences must be a DataFrame")

    if "User_ID" not in df.columns or "Ranked_Items" not in df.columns:
        raise ValueError("preferences must contain User_ID and Ranked_Items")

    # Canonical form
    canon = (
        df[["User_ID", "Ranked_Items"]]
        .copy()
        .sort_values("User_ID", kind="mergesort")
        .reset_index(drop=True)
    )
    # Ensure lists become tuples (hashable, stable)
    canon["Ranked_Items"] = canon["Ranked_Items"].apply(lambda x: tuple(x))

    # Hash via pandas stable serialization
    payload = canon.to_csv(index=False).encode("utf-8")
    return hashlib.sha256(payload).hexdigest()

def test_multiple_sampled_rankings_differ(sampled_rankings_paths):
    """
    Sanity check: sampled_rankings across samples should not all be identical.
    Strong check: no two files have identical fingerprints.

    If you EXPECT occasional collisions, relax to:
      assert len(set(fps)) >= 2
    """
    fps = []
    for p in sampled_rankings_paths:
        prefs = load_sampled_preferences(p)
        prefs = prefs.astype({"User_ID": prefs["User_ID"].dtype})  # no-op, but explicit
        fps.append((_df_fingerprint_preferences(prefs), p))

    # detect collisions
    seen = {}
    collisions = []
    for fp, path in fps:
        if fp in seen:
            collisions.append((seen[fp], path))
        else:
            seen[fp] = path

    assert not collisions, (
        "Found identical sampled_rankings across different samples (possible copy/overwrite bug).\n"
        + "\n".join([f"{a} == {b}" for a, b in collisions])
    )

def test_consensus_files_exist(consensus_dir):
    paths = sorted(glob.glob(os.path.join(consensus_dir, "*.txt")))
    assert len(paths) > 0, f"No .txt files found in {consensus_dir}"

def test_load_consensus_ranking_parses_nonempty(consensus_dir):
    paths = sorted(glob.glob(os.path.join(consensus_dir, "*.txt")))
    # pick first file
    committee = load_consensus_ranking(paths[0])
    assert isinstance(committee, list)
    assert len(committee) > 0, f"Parsed empty committee from {paths[0]}"

def test_committees_are_ints_and_unique(consensus_dir):
    _, committees = load_all_committees(consensus_dir)

    for name, committee in committees.items():
        assert isinstance(committee, list), f"{name} committee is not a list"
        assert len(committee) > 0, f"{name} committee empty"

        # all ints
        assert all(isinstance(x, int) for x in committee), f"{name} has non-int entries"

        # no duplicates inside a committee
        assert len(set(committee)) == len(committee), f"{name} committee contains duplicates"

def test_consensus_text_files_not_byte_identical(consensus_dir):
    """
    Strong check: files should not be identical at the byte level.
    This catches accidental copy/overwrite issues.
    """
    paths = sorted(glob.glob(os.path.join(consensus_dir, "*.txt")))
    hashes = {p: sha256_file(p) for p in paths}

    unique_hashes = set(hashes.values())
    assert len(unique_hashes) == len(paths), (
        "Some consensus .txt files are byte-identical (possible overwrite/copy bug). "
        "Identical groups:\n"
        + "\n".join(
            f"{h}: {[os.path.basename(p) for p,v in hashes.items() if v==h]}"
            for h in unique_hashes
            if sum(1 for v in hashes.values() if v == h) > 1
        )
    )

def test_most_committees_differ_at_topk(consensus_dir):
    """
    Practical sanity: methods should differ in the first K items.
    This is often what you care about for prefix evaluation.
    We require at least 2 distinct top-K sequences.
    """
    TOPK = int(os.environ.get("TOPK", "20"))
    _, committees = load_all_committees(consensus_dir)

    topk_sigs = set()
    for method, committee in committees.items():
        topk = tuple(committee[:TOPK])
        topk_sigs.add(topk)

    assert len(topk_sigs) >= 2, (
        f"All methods share the same top-{TOPK} ordering. "
        "This is suspicious unless you expect identical rankings."
    )

def test_consensus_methods_cover_expected_count(consensus_dir):
    """
    Optional: if you expect a minimum number of methods, enforce it.
    """
    MIN_METHODS = int(os.environ.get("MIN_METHODS", "5"))
    paths = sorted(glob.glob(os.path.join(consensus_dir, "*.txt")))
    assert len(paths) >= MIN_METHODS, (
        f"Expected at least {MIN_METHODS} consensus files, found {len(paths)}."
    )
