# policy_analyzer_with_logging.py

import yaml
import json
import pandas as pd
import numpy as np
from sentence_transformers import SentenceTransformer
from sklearn.metrics.pairwise import cosine_similarity
import logging

# Set up a global logger
logger = logging.getLogger(__name__)


def load_config(config_path='config.yml'):
    """Loads the main configuration and weights from YAML files."""
    try:
        logger.info(f"Attempting to load main config from '{config_path}'...")
        with open(config_path, 'r', encoding='utf-8') as f:
            config = yaml.safe_load(f)

        weights_path = config.get('paths', {}).get('weights_file', 'weights.yml')
        logger.info(f"Attempting to load weights from '{weights_path}'...")
        with open(weights_path, 'r', encoding='utf-8') as f:
            weights = yaml.safe_load(f)

        config['weights'] = weights
        logger.info("✅ Configuration and weights loaded successfully.")
        return config

    except FileNotFoundError as e:
        logger.error(f"❌ Configuration file not found -> {e}")
        return None
    except Exception as e:
        logger.error(f"❌ Error loading configuration: {e}")
        return None


def flatten_json(data, parent_key='', sep='.'):
    """Flattens a nested JSON/dictionary."""
    items = {}
    for k, v in data.items():
        new_key = parent_key + sep + k if parent_key else k
        if isinstance(v, dict):
            items.update(flatten_json(v, new_key, sep=sep))
        else:
            items[new_key] = str(v)
    return items


def calculate_jaccard_similarity(str1, str2):
    """Calculates Jaccard similarity between two strings treated as sets of words."""
    set1 = set(str(str1).lower().replace(',', ' ').split())
    set2 = set(str(str2).lower().replace(',', ' ').split())
    intersection = len(set1.intersection(set2))
    union = len(set1.union(set2))
    return intersection / union if union > 0 else 0.0


def compare_policy_to_rules(policy_flat, rules_df, config, embedding_model):
    """
    Compares a flattened policy against all rules using the Hybrid Model approach.
    """
    weights = config['weights']
    thresholds = config['thresholds']

    # Pre-calculate embeddings for all relevant policy attributes to avoid re-computation
    logger.info("Pre-calculating embeddings for policy attributes...")
    policy_embeddings = {}
    for attr in weights['should_match']:
        if attr in policy_flat:
            policy_embeddings[attr] = embedding_model.encode(policy_flat[attr])

    candidate_rules = {}
    grouped_rules = rules_df.groupby('brid')
    total_rules = len(grouped_rules)

    logger.info(f"\n--- Stage 2: Filtering & Scoring ---")
    logger.info(f"🔎 Analyzing policy against {total_rules} rules...")

    for i, (rule_id, group) in enumerate(grouped_rules):
        logger.info(f"[{i + 1}/{total_rules}] Processing Rule ID: '{rule_id}'")
        rule_attrs = dict(zip(group['attribute_path'], group['attribute_value']))

        # --- Filtering Stage ('must_match') ---
        is_candidate = True
        for attr in weights['must_match']:
            policy_val = policy_flat.get(attr)
            rule_val = rule_attrs.get(attr)
            if policy_val != rule_val:
                logger.warning(
                    f"  -> Discarded. Reason: 'must_match' failed on '{attr}'. (Policy: '{policy_val}' != Rule: '{rule_val}')")
                is_candidate = False
                break

        if not is_candidate:
            continue  # Discard rule and move to the next one

        logger.info(f"  -> PASSED filter. Now scoring as a candidate.")

        # --- Scoring Stage ('should_match') for Candidates ---
        total_score = 0.0
        total_possible_weight = 0.0
        match_details = []

        for attr, weight in weights['should_match'].items():
            total_possible_weight += weight
            policy_value = policy_flat.get(attr)
            rule_value = rule_attrs.get(attr)

            if policy_value and rule_value:
                # Calculate similarity scores
                jaccard_score = calculate_jaccard_similarity(policy_value, rule_value)
                policy_vec = policy_embeddings.get(attr)
                rule_vec = embedding_model.encode(rule_value)
                cosine_score = cosine_similarity([policy_vec], [rule_vec])[0][0]

                attr_score = 0.0
                if policy_value == rule_value:
                    attr_score = 1.0
                elif cosine_score >= thresholds['cosine_similarity']:
                    attr_score = cosine_score
                elif jaccard_score >= thresholds['jaccard_similarity']:
                    attr_score = jaccard_score

                weighted_score = attr_score * weight
                total_score += weighted_score

                match_details.append({"attribute": attr, "score": round(attr_score, 4)})

        final_score = total_score / total_possible_weight if total_possible_weight > 0 else 0.0
        logger.info(f"  -> Calculated final score: {final_score:.4f}")

        candidate_rules[rule_id] = {"final_score": round(final_score, 4), "match_details": match_details}

    # --- Stage 3: Final Analysis & Classification ---
    logger.info("\n--- Stage 3: Final Analysis ---")
    if not candidate_rules:
        logger.warning("No candidate rules passed the filter.")
        return {"classification": "NEW", "best_match": None, "top_matches": []}

    sorted_candidates = sorted(candidate_rules.items(), key=lambda item: item[1]['final_score'], reverse=True)
    best_rule_id, best_match_details = sorted_candidates[0]

    logger.info(
        f"Found {len(candidate_rules)} candidate(s). Best match is Rule '{best_rule_id}' with score {best_match_details['final_score']}.")

    classification = "UPDATE" if best_match_details['final_score'] >= thresholds['classification'] else "NEW"
    logger.info(f"Classification threshold is {thresholds['classification']}. Final decision: '{classification}'")

    return {
        "classification": classification,
        "best_match": {"rule_id": best_rule_id, "score": best_match_details['final_score']},
        "top_matches": [{item[0]: item[1]} for item in sorted_candidates[:5]]
    }


def main():
    """Main execution function to run the entire analysis pipeline."""
    # Configure the logger to show timestamp, level, and message
    logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(levelname)s - %(message)s')

    logger.info("🚀 Starting Policy Analyzer Script...")

    # 1. Load Configurations
    logger.info("\n--- Stage 1: Loading Data & Models ---")
    config = load_config()
    if not config:
        return

    # 2. Load and Prepare Data
    try:
        rules_df = pd.read_csv(config['paths']['rules_csv'])
        with open(config['paths']['policy_json'], 'r', encoding='utf-8') as f:
            policy_data = json.load(f)

        policy_flat = flatten_json(policy_data)
        logger.info(f"✅ Policy and rules data loaded. Found {len(rules_df.brid.unique())} unique rules.")
    except FileNotFoundError as e:
        logger.error(f"❌ Data file not found -> {e}")
        return

    # 3. Initialize AI Model
    logger.info(f"🤖 Loading embedding model ('{config['models']['embedding_model']}'). This may take a moment...")
    model = SentenceTransformer(config['models']['embedding_model'])
    logger.info("🤖 Model loaded successfully.")

    # 4. Run Comparison
    results = compare_policy_to_rules(policy_flat, rules_df, config, model)

    # 5. Save and Display Results
    output_path = config['paths']['output_json']
    with open(output_path, 'w') as f:
        json.dump(results, f, indent=2)

    logger.info(f"\n🏁 Analysis Complete. Results saved to '{output_path}'")


if __name__ == "__main__":
    main()