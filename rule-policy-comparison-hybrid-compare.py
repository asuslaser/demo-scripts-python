import json
import pandas as pd
# We use 'thefuzz' as it's the maintained fork of 'fuzzywuzzy'
# You may need to install it: pip install pandas thefuzz
from thefuzz import fuzz
import os
import pprint
import ast  # For safely parsing string-lists
import logging
from collections import OrderedDict  # <-- 1. IMPORT OrderedDict

# --- Configuration ---

# Set up detailed logging
# Using print() for simplicity and direct console output as requested by user
LOG_LEVEL = "DEBUG"  # Set to "INFO" for less detail


def log_debug(message):
    if LOG_LEVEL == "DEBUG":
        print(f"[DEBUG] {message}")


def log_info(message):
    print(f"[INFO] {message}")


# The minimum similarity score for a string to be "similar" (Count1_3)
# This is your ">70%" rule
SIMILARITY_VALUE_THRESHOLD = 70

# The minimum composite score to be considered an "Update"
# This is your "> 80%" rule
UPDATE_CLASSIFICATION_THRESHOLD = 0.80

# The flattened attribute path for the Payor.
# This is CRITICAL for the "Update" logic.
PAYOR_ATTRIBUTE_PATH = "rule.payer.name"  # Example: "payor.name" or "rule.payor.id"


def _flatten_helper(obj, path=''):
    """
    Recursively flattens a nested dictionary or list.
    Uses OrderedDict to maintain key order.
    """
    flat_dict = OrderedDict()  # <-- 2. USE OrderedDict here
    if isinstance(obj, OrderedDict):  # Check for OrderedDict first
        for k, v in obj.items():
            # Create a new path by joining with '.'
            new_path = f"{path}.{k}" if path else k
            flat_dict.update(_flatten_helper(v, new_path))
    elif isinstance(obj, list):
        # Check if it's a list of non-dict/non-list items
        # If so, store it as a string representation of the list
        if all(not isinstance(item, (dict, list)) for item in obj):
            log_debug(f"Flattening list at path '{path}' to string.")
            flat_dict[path] = str(obj)
        else:
            # Otherwise, keep flattening recursively
            log_debug(f"Recursively flattening list at path '{path}'.")
            for i, v in enumerate(obj):
                new_path = f"{path}.{i}" if path else str(i)
                flat_dict.update(_flatten_helper(v, new_path))
    else:
        # Base case: we have a primitive value.
        # Standardize all values to strings for comparison.
        if obj is None:
            flat_dict[path] = "null"
        elif isinstance(obj, bool):
            flat_dict[path] = str(obj).lower()
        else:
            flat_dict[path] = str(obj)
    return flat_dict


def flatten_policy_json(json_path):
    """
    Reads a JSON file and flattens it into a single-level dictionary,
    preserving the order of keys from the file.
    """
    log_info(f"--- Step 1: Flattening Policy JSON: {json_path} ---")
    try:
        with open(json_path, 'r', encoding='utf-8') as f:
            # Use object_pairs_hook=OrderedDict to load JSON in order
            data = json.load(f, object_pairs_hook=OrderedDict)

        flat_policy = _flatten_helper(data)
        log_info(f"Policy flattened into {len(flat_policy)} attributes.")
        log_debug("Flattened Policy Content:")
        log_debug(pprint.pformat(flat_policy))
        return flat_policy
    except Exception as e:
        log_info(f"Error reading or flattening {json_path}: {e}")
        return None


def build_rules_database(csv_path):
    """
    Reads the "long" rules CSV and converts it into an optimized
    in-memory dictionary grouped by rule_id.

    Expected CSV columns: rule_id, attribute_path, attribute_value
    """
    log_info(f"--- Step 2: Building Rules Database from: {csv_path} ---")
    rules_db = {}
    try:
        # Read all columns as strings to ensure consistency
        df = pd.read_csv(csv_path, dtype=str).fillna("null")

        # We need rule_id, attribute_path, and attribute_value.
        required_cols = {'rule_id', 'attribute_path', 'attribute_value'}
        if not required_cols.issubset(df.columns):
            log_info(f"Error: CSV is missing one or more required columns: {required_cols - set(df.columns)}")
            return {}

        # This is the core optimization.
        for rule_id, group in df.groupby('rule_id'):
            # Create the dictionary of attributes
            attributes = dict(zip(group.attribute_path, group.attribute_value))

            rules_db[rule_id] = {
                "attrs": attributes
            }
            log_debug(f"Loaded Rule '{rule_id}' with {len(attributes)} attributes.")

    except Exception as e:
        log_info(f"Error building rules database from CSV: {e}")
        return {}

    log_info(f"Rules database built. {len(rules_db)} unique rules loaded.")
    return rules_db


def _get_jaccard_score(str_list_a, str_list_b):
    """
    Helper function to safely parse two string-lists and get their
    Jaccard similarity score.
    """
    log_debug(f"  [Jaccard] Parsing List A: {str_list_a}")
    log_debug(f"  [Jaccard] Parsing List B: {str_list_b}")
    try:
        # ast.literal_eval is a safe way to parse Python literals
        list_a = ast.literal_eval(str_list_a)
        list_b = ast.literal_eval(str_list_b)

        # Ensure they are actually lists (could be "null" or "123")
        if not isinstance(list_a, list) or not isinstance(list_b, list):
            log_debug("  [Jaccard] One or both values is not a list. Score: 0.0")
            return 0.0

        # Convert to sets for Jaccard calculation
        set_a = {str(item) for item in list_a if item is not None}
        set_b = {str(item) for item in list_b if item is not None}

        log_debug(f"  [Jaccard] Set A ({len(set_a)} items): {set_a}")
        log_debug(f"  [Jaccard] Set B ({len(set_b)} items): {set_b}")

        if not set_a and not set_b:
            log_debug("  [Jaccard] Both sets are empty. Score: 1.0")
            return 1.0  # Both are empty, so they are identical

        intersection = set_a.intersection(set_b)
        union = set_a.union(set_b)

        log_debug(f"  [Jaccard] Intersection ({len(intersection)} items): {intersection}")
        log_debug(f"  [Jaccard] Union ({len(union)} items): {union}")

        if not union:
            log_debug("  [Jaccard] Union is empty. Score: 0.0")
            return 0.0  # Avoid division by zero if union is empty

        score = len(intersection) / len(union)
        log_debug(f"  [Jaccard] Final Score (Intersection/Union): {score:.4f}")
        return score

    except (ValueError, SyntaxError, TypeError) as e:
        log_debug(f"  [Jaccard] Error parsing string-list: {e}. Score: 0.0")
        return 0.0


def calculate_similarity(policy_flat, rule_id, rule_flat):
    """
    Calculates the similarity score between the policy and a single rule
    using a HYBRID formula (Jaccard for codes, Fuzz for text).
    """
    log_info(f"\n--- Comparing Policy vs. Rule '{rule_id}' ---")

    p_keys = set(policy_flat.keys())
    r_keys = set(rule_flat.keys())

    # Count1: Similar attributes (keys) used in both
    common_keys = p_keys.intersection(r_keys)
    count1 = len(common_keys)
    log_debug(f"Count1 (Common Attributes): {count1}")
    log_debug(f"  {common_keys}")

    # Count2: Attributes used only in policy
    policy_only_keys = p_keys.difference(r_keys)
    count2 = len(policy_only_keys)
    log_debug(f"Count2 (Policy-Only Attributes): {count2}")
    log_debug(f"  {policy_only_keys}")

    # Count3: Attributes used only in rule
    rule_only_keys = r_keys.difference(p_keys)
    count3 = len(rule_only_keys)
    log_debug(f"Count3 (Rule-Only Attributes): {count3}")
    log_debug(f"  {rule_only_keys}")

    # This is the denominator of your formula
    denominator = count1 + count2 + count3
    log_debug(f"Total Unique Attributes (Denominator): {denominator}")

    # Avoid division by zero
    if denominator == 0:
        score = 1.0 if not p_keys and not r_keys else 0.0
        log_info(f"Score for '{rule_id}': {score} (Denominator is zero)")
        return score

    count1_2 = 0  # Exact value match
    count1_3 = 0  # Similarity value match (>70%)

    log_debug("--- Checking Common Attributes for Value Matches ---")
    for key in common_keys:
        v_policy = policy_flat[key]
        v_rule = rule_flat[key]

        log_debug(f"  Checking key: '{key}'")
        log_debug(f"    Policy Value: {v_policy}")
        log_debug(f"    Rule Value:   {v_rule}")

        # 1. Check for exact value match
        if v_policy == v_rule:
            count1_2 += 1
            log_debug("    -> Result: EXACT MATCH")
        else:
            # 2. If not exact, use HYBRID similarity logic
            similarity_score_percent = 0.0

            if 'procedureCodes' in key or 'diagnosisCodes' in key:
                log_debug("    -> Type: Code List. Using Jaccard Similarity...")
                similarity_score = _get_jaccard_score(v_policy, v_rule)
                similarity_score_percent = similarity_score * 100

            else:
                log_debug("    -> Type: Text. Using TheFuzz (token_set_ratio)...")
                similarity_score_percent = fuzz.token_set_ratio(v_policy, v_rule)

            log_debug(f"    -> Similarity Score: {similarity_score_percent:.2f}%")

            if similarity_score_percent > SIMILARITY_VALUE_THRESHOLD:
                count1_3 += 1
                log_debug(f"    -> Result: SIMILARITY MATCH (Score > {SIMILARITY_VALUE_THRESHOLD}%)")
            else:
                log_debug("    -> Result: NO MATCH")

    log_debug("--- Final Score Calculation ---")
    log_debug(f"Count1_2 (Exact Value Matches): {count1_2}")
    log_debug(f"Count1_3 (Similar Value Matches): {count1_3}")

    # This is the numerator of your formula
    numerator = count1_2 + count1_3
    log_debug(f"Numerator (Count1_2 + Count1_3): {numerator}")

    score = numerator / denominator
    log_info(f"Score for '{rule_id}': {numerator} / {denominator} = {score:.4f}")
    return score


def process_policy_comparison(policy_json_path, rules_csv_path):
    """
    Main function to run the entire comparison and classification process.
    """

    # 1. Load and flatten the new policy
    policy_flat = flatten_policy_json(policy_json_path)
    if not policy_flat:
        return {"error": "Could not process policy JSON."}

    # 2. Build the optimized in-memory rules database
    rules_db = build_rules_database(rules_csv_path)
    if not rules_db:
        return {"error": "Could not build rules database from CSV."}

    # 3. Compare policy against EVERY rule to find the best match
    all_scores = {}
    log_info(f"\n--- Step 3: Comparing Policy vs. All {len(rules_db)} Rules ---")
    for rule_id, rule_data in rules_db.items():
        rule_flat = rule_data["attrs"]
        score = calculate_similarity(policy_flat, rule_id, rule_flat)
        all_scores[rule_id] = score

    if not all_scores:
        log_info("No rules to compare against.")
        return {"error": "No rules found in database."}

    # 4. Find the best matching rule
    log_info("\n--- Step 4: Finding Best Match ---")
    best_match_id = max(all_scores, key=all_scores.get)
    best_score = all_scores[best_match_id]

    log_info(f"Comparison complete. Best match is Rule '{best_match_id}' with score: {best_score:.4f}")

    # 5. Apply Classification Logic (Update vs. New)
    log_info("\n--- Step 5: Classifying Result ---")
    classification = "NEW"

    policy_payor = policy_flat.get(PAYOR_ATTRIBUTE_PATH)

    # Get the data for the best-matching rule
    best_rule_data = rules_db[best_match_id]
    best_rule_payor = best_rule_data["attrs"].get(PAYOR_ATTRIBUTE_PATH)

    log_info(f"Policy Payor: {policy_payor}")
    log_info(f"Best Match '{best_match_id}' Payor: {best_rule_payor}")

    # --- Your "Update" Logic ---
    if best_score > UPDATE_CLASSIFICATION_THRESHOLD and policy_payor == best_rule_payor:
        classification = "UPDATE"
        log_info(f"-> CLASSIFICATION: UPDATE")
        log_info(f"  Reason: Score ({best_score:.4f}) > {UPDATE_CLASSIFICATION_THRESHOLD} AND Payor matches.")
    else:
        # --- Your "New" Logic ---
        classification = "NEW"
        log_info(f"-> CLASSIFICATION: NEW")
        if not (best_score > UPDATE_CLASSIFICATION_THRESHOLD):
            log_info(f"  Reason: Score ({best_score:.4f}) is not > {UPDATE_CLASSIFICATION_THRESHOLD}")
        elif policy_payor != best_rule_payor:
            log_info(f"  Reason: Score was high, but Payor does not match ('{policy_payor}' vs '{best_rule_payor}')")

    # 6. Format the final output
    final_result = {
        "policy_file": policy_json_path,
        "classification": classification,
        "best_match_rule_id": best_match_id,
        "best_match_score": best_score,
        "payor_match": (policy_payor == best_rule_payor) if policy_payor else False,
        "all_rule_scores": all_scores  # Optionally include all scores
    }

    log_info("\n--- Step 6: Final Result ---")
    return final_result


# --- Example of how to run this script ---
if __name__ == "__main__":

    log_info("--- Starting Policy Comparison Runner (Verbose Logic) ---")

    # --- *** SET YOUR FILE PATHS HERE *** ---
    # Update these paths to point to your actual files.
    # Use raw strings (r"...") or double backslashes (\\) for Windows paths.

    # Path to your main CSV containing all rule attributes
    RULES_CSV_FILE_PATH = r"D:\Proj Docs\policy analyzer\rules.csv"

    # Path to the first policy JSON you want to test
    POLICY_1_JSON_PATH = r"D:\Proj Docs\policy analyzer\policy_sample.json"

    # Path to the second policy JSON you want to test
    POLICY_2_JSON_PATH = r"D:\Proj Docs\policy analyzer\policy_sample.json"

    # --- *** END OF FILE PATHS *** ---

    # Check if files exist before running
    if not os.path.exists(RULES_CSV_FILE_PATH):
        log_info(f"Error: Rules CSV file not found at: {RULES_CSV_FILE_PATH}")
    elif not os.path.exists(POLICY_1_JSON_PATH):
        log_info(f"Error: Policy 1 JSON file not found at: {POLICY_1_JSON_PATH}")
    elif not os.path.exists(POLICY_2_JSON_PATH):
        log_info(f"Error: Policy 2 JSON file not found at: {POLICY_2_JSON_PATH}")
    else:
        # 1. Run the process for the first policy
        print("\n\n" + "=" * 50)
        log_info(f"--- TEST CASE 1: Processing Policy '{POLICY_1_JSON_PATH}' ---")
        print("=" * 50)
        result_1 = process_policy_comparison(
            policy_json_path=POLICY_1_JSON_PATH,
            rules_csv_path=RULES_CSV_FILE_PATH
        )
        pprint.pprint(result_1)

        # 2. Run the process for the second policy
        print("\n\n" + "=" * 50)
        log_info(f"--- TEST CASE 2: Processing Policy '{POLICY_2_JSON_PATH}' ---")
        print("=" * 50)
        result_2 = process_policy_comparison(
            policy_json_path=POLICY_2_JSON_PATH,
            rules_csv_path=RULES_CSV_FILE_PATH
        )
        pprint.pprint(result_2)



