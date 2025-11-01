import os
import json
import csv
import pandas as pd
import string_similarity  # Our other .py file
from pathlib import Path


def load_config(path="config.json") -> dict:
    # ... (no change to this function) ...
    with open(path, 'r') as f:
        return json.load(f)


def flatten_json(y: dict) -> dict:
    """Recursively flattens a nested JSON/dict."""
    flat_map = {}

    def _flatten(obj, prefix=''):
        if isinstance(obj, dict):
            if not obj:  # Handle empty dict
                flat_map[prefix.rstrip('.')] = None
            for k, v in obj.items():
                _flatten(v, f"{prefix}{k}.")
        elif isinstance(obj, list):
            if not obj:  # Handle empty list
                flat_map[prefix.rstrip('.')] = None
            for i, item in enumerate(obj):
                _flatten(item, f"{prefix}{i}.")
        else:
            # Store the value, stripping the last '.' from the key
            # --- THIS IS THE FIX ---
            # Original bug: flat_map[prefix.rstrip('.')] = v
            flat_map[prefix.rstrip('.')] = obj
            # --- END FIX ---

    _flatten(y)
    return flat_map


def load_rules_database(csv_path: str) -> dict:
    # ... (no change to this function) ...
    print(f"Loading and grouping rules from {csv_path}...")
    rules_db = {}
    try:
        df = pd.read_csv(csv_path, dtype=str)
        # Handle potential empty CSV
        if df.empty:
            print("Warning: Rules CSV is empty.")
            return {}

        # Drop rows where 'rule_id' or 'attribute_path' is missing
        df = df.dropna(subset=['rule_id', 'attribute_path'])

        for index, row in df.iterrows():
            rule_id = row['rule_id']
            attr_path = row['attribute_path']
            attr_value = row['attribute_value']

            if rule_id not in rules_db:
                rules_db[rule_id] = {}

            # Assuming one value per path for simplicity
            # If duplicates, last one wins.
            rules_db[rule_id][attr_path] = attr_value

        print(f"Successfully loaded and grouped {len(rules_db)} unique rules.")

    except FileNotFoundError:
        print(f"Error: Rules CSV file not found at {csv_path}")
    except pd.errors.EmptyDataError:
        print(f"Error: Rules CSV file is empty at {csv_path}")
    except Exception as e:
        print(f"An error occurred while loading rules: {e}")

    return rules_db


def compare_policy_to_rules(policy_map: dict, policy_filename: str, rules_db: dict, config: dict) -> (dict, list):
    # ... (no change to this function) ...
    scores = {}
    trace_log = []  # This will store our detailed trace data

    cfg_logic = config.get("LOGIC_CONTROLS", {})
    cfg_thresholds = config.get("MATCH_THRESHOLDS", {})

    long_str_limit = cfg_logic.get("LONG_STRING_CHAR_LIMIT", 50)
    fuzzy_thresh = cfg_thresholds.get("FUZZY_STRING_THRESHOLD", 0.9)
    jaccard_thresh = cfg_thresholds.get("LONG_STRING_JACCARD_THRESHOLD", 0.9)

    for rule_id, rule_map in rules_db.items():
        policy_keys = set(policy_map.keys())
        rule_keys = set(rule_map.keys())

        # Calculate key sets
        count1_keys = policy_keys.intersection(rule_keys)
        count2_keys = policy_keys.difference(rule_keys)
        count3_keys = rule_keys.difference(policy_keys)

        count1 = len(count1_keys)
        count2 = len(count2_keys)
        count3 = len(count3_keys)

        count1_2 = 0  # Exact match
        count1_3 = 0  # Fuzzy match

        # --- Process Count 1 (Common Attributes) ---
        for key in count1_keys:
            policy_value_raw = policy_map.get(key)
            rule_value_raw = rule_map.get(key)

            policy_val_norm = string_similarity.normalize(policy_value_raw)
            rule_val_norm = string_similarity.normalize(rule_value_raw)

            match_type = "Mismatch"
            contribution = 0

            # 1. Check for Exact Match
            if policy_val_norm == rule_val_norm:
                count1_2 += 1
                match_type = "Exact"
                contribution = 1
            else:
                # 2. Check for Fuzzy Match (if not exact)
                is_fuzzy_match = False
                # Use Jaro-Winkler for short strings
                if len(policy_val_norm) < long_str_limit and len(rule_val_norm) < long_str_limit:
                    if string_similarity.jaro_winkler_match(policy_val_norm, rule_val_norm, fuzzy_thresh):
                        is_fuzzy_match = True
                # Use Jaccard-on-Words for long strings
                else:
                    if string_similarity.jaccard_on_words_match(policy_val_norm, rule_val_norm, jaccard_thresh):
                        is_fuzzy_match = True

                if is_fuzzy_match:
                    count1_3 += 1
                    match_type = "Fuzzy"
                    contribution = 1

            # Add detailed log entry for this common attribute
            trace_log.append({
                "policy_file": policy_filename,
                "rule_id": rule_id,
                "attribute_path": key,
                "policy_value": policy_value_raw,
                "rule_value": rule_value_raw,
                "comparison_type": "Common (Count1)",
                "match_type": match_type,
                "score_contribution": contribution
            })

        # --- Process Count 2 (Policy-Only Attributes) ---
        for key in count2_keys:
            trace_log.append({
                "policy_file": policy_filename,
                "rule_id": rule_id,
                "attribute_path": key,
                "policy_value": policy_map.get(key),
                "rule_value": None,
                "comparison_type": "Policy-Only (Count2)",
                "match_type": "N/A",
                "score_contribution": 0
            })

        # --- Process Count 3 (Rule-Only Attributes) ---
        for key in count3_keys:
            trace_log.append({
                "policy_file": policy_filename,
                "rule_id": rule_id,
                "attribute_path": key,
                "policy_value": None,
                "rule_value": rule_map.get(key),
                "comparison_type": "Rule-Only (Count3)",
                "match_type": "N/A",
                "score_contribution": 0
            })

        # Calculate final similarity score for this rule
        denominator = count1 + count2 + count3
        score = 0.0
        if denominator > 0:
            score = (count1_2 + count1_3) / denominator

        scores[rule_id] = score

    return scores, trace_log


def get_final_decision(policy_map: dict, rules_db: dict, scores: dict, config: dict) -> (str, str, float):
    # ... (no change to this function) ...
    if not scores:
        return "New", "N/A", 0.0

    # Find the best matching rule
    best_rule_id = max(scores, key=scores.get)
    max_score = scores[best_rule_id]

    main_threshold = config.get("MATCH_THRESHOLDS", {}).get("MAIN_SIMILARITY_THRESHOLD", 0.8)

    # Default decision
    decision = "New"

    if max_score >= main_threshold:
        # If score is high, check payor match before declaring "Update"
        cfg_logic = config.get("LOGIC_CONTROLS", {})
        payor_attr = cfg_logic.get("PAYOR_MATCH_ATTRIBUTE")
        payor_thresh = cfg_logic.get("PAYOR_MATCH_THRESHOLD", 0.9)

        policy_payor = policy_map.get(payor_attr)
        best_rule_map = rules_db.get(best_rule_id, {})
        best_rule_payor = best_rule_map.get(payor_attr)

        # Normalize payor strings
        policy_payor_norm = string_similarity.normalize(policy_payor)
        rule_payor_norm = string_similarity.normalize(best_rule_payor)

        payor_is_match = False
        if policy_payor_norm == rule_payor_norm:
            payor_is_match = True
        elif string_similarity.jaro_winkler_match(policy_payor_norm, rule_payor_norm, payor_thresh):
            payor_is_match = True

        if payor_is_match:
            decision = "Update"
        else:
            decision = "New (Payor Mismatch)"

    return decision, best_rule_id, max_score


def main():
    # ... (no change to this function) ...
    config = load_config("config.json")
    paths = config.get("FILE_PATHS", {})

    # Ensure output directory exists
    output_dir = Path(paths.get("OUTPUT_CSV_PATH", "output/results.csv")).parent
    output_dir.mkdir(parents=True, exist_ok=True)

    # Load rules once
    rules_db = load_rules_database(paths.get("RULES_CSV_PATH"))
    if not rules_db:
        print("Halting execution: Rules database is empty or failed to load.")
        return

    policy_folder = paths.get("POLICY_JSON_FOLDER")
    print(f"\nProcessing policy JSON files from {policy_folder}...")

    policy_files = [f for f in os.listdir(policy_folder) if f.endswith('.json')]
    if not policy_files:
        print(f"No .json files found in {policy_folder}")
        return

    final_results = []

    for policy_filename in policy_files:
        policy_filepath = os.path.join(policy_folder, policy_filename)

        try:
            with open(policy_filepath, 'r') as f:
                policy_json = json.load(f)

            # Flatten the policy JSON
            policy_map = flatten_json(policy_json)

            # Compare policy to all rules and get trace data
            scores, trace_data = compare_policy_to_rules(policy_map, policy_filename, rules_db, config)

            # Get final decision
            decision, best_rule_id, max_score = get_final_decision(policy_map, rules_db, scores, config)

            # --- Save Trace CSV ---
            if trace_data:
                trace_filename = f"trace_{Path(policy_filename).stem}.csv"
                trace_csv_path = output_dir / trace_filename

                trace_df = pd.DataFrame(trace_data)
                # Reorder columns for readability
                cols = ["policy_file", "rule_id", "comparison_type", "attribute_path",
                        "policy_value", "rule_value", "match_type", "score_contribution"]
                trace_df = trace_df[cols]

                # Sort for easier analysis
                trace_df = trace_df.sort_values(by=["rule_id", "comparison_type", "attribute_path"])
                trace_df.to_csv(trace_csv_path, index=False, quoting=csv.QUOTE_ALL)

                print(
                    f"  - Processed {policy_filename}: Decision={decision}, BestMatch={best_rule_id}, Score={max_score * 100:.2f}%")
                print(f"    -> Detailed trace saved to {trace_csv_path}")
            else:
                print(f"  - Processed {policy_filename}: No comparisons made (empty trace).")

            final_results.append({
                "policy_file": policy_filename,
                "decision": decision,
                "best_match_rule_id": best_rule_id,
                "similarity_score": f"{max_score * 100:.2f}%"
            })

        except json.JSONDecodeError:
            print(f"  - Error processing {policy_filename}: Invalid JSON.")
            final_results.append({
                "policy_file": policy_filename,
                "decision": "Error",
                "best_match_rule_id": "N/A",
                "similarity_score": "N/A"
            })
        except Exception as e:
            print(f"  - An unexpected error occurred with {policy_filename}: {e}")
            final_results.append({
                "policy_file": policy_filename,
                "decision": "Error",
                "best_match_rule_id": "N/A",
                "similarity_score": "N/A"
            })

    # Save final summary results
    summary_csv_path = paths.get("OUTPUT_CSV_PATH", "output/results.csv")
    if final_results:
        summary_df = pd.DataFrame(final_results)
        summary_df.to_csv(summary_csv_path, index=False)

    print(f"\nAll processing complete. Summary results saved to {summary_csv_path}")


if __name__ == "__main__":
    main()

