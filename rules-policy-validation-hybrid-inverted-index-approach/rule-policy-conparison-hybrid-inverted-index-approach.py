#pip install pyyaml pandas numpy scikit-learn sentence-transformers

# policy_analyzer_inverted_index.py

import yaml
import json
import pandas as pd
from sentence_transformers import SentenceTransformer
from sklearn.metrics.pairwise import cosine_similarity
import logging
from collections import Counter

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


def build_inverted_index(rules_df):
    """Builds an inverted index from the rules DataFrame."""
    logger.info("Building inverted index from rules data...")
    index = {}
    for _, row in rules_df.iterrows():
        rule_id, value = row['id'], str(row['value'])
        if value not in index:
            index[value] = []
        index[value].append(rule_id)
    logger.info(f"✅ Inverted index built successfully with {len(index)} unique values.")
    return index


def get_candidates_from_index(policy_flat, inverted_index, top_n):
    """Uses the inverted index to find the most relevant candidate rule IDs."""
    logger.info("\n--- Stage 2: Candidate Retrieval from Index ---")
    candidate_scores = Counter()
    for attr, value in policy_flat.items():
        if value in inverted_index:
            matched_rule_ids = inverted_index[value]
            logger.info(f"  -> Hit found for value '{value}' in rules: {matched_rule_ids}")
            candidate_scores.update(matched_rule_ids)

    if not candidate_scores:
        logger.warning("No candidate rules found from index lookup.")
        return []

    # Get the most common rule IDs as top candidates
    top_candidates = [rule_id for rule_id, count in candidate_scores.most_common(top_n)]
    logger.info(f"✅ Retrieved top {len(top_candidates)} candidate(s): {top_candidates}")
    return top_candidates


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


def rank_candidates(policy_flat, rules_df, candidate_ids, config, embedding_model):
    """Performs detailed scoring on a small list of candidate rules."""
    logger.info("\n--- Stage 3: Detailed Ranking of Candidates ---")
    weights = config['weights']
    thresholds = config['thresholds']

    policy_embeddings = {attr: embedding_model.encode(policy_flat[attr])
                         for attr in weights['should_match'] if attr in policy_flat}

    final_results = {}
    grouped_rules = rules_df.groupby('id')

    for rule_id in candidate_ids:
        logger.info(f"Scoring candidate Rule ID: '{rule_id}'")
        group = grouped_rules.get_group(rule_id)
        rule_attrs = dict(zip(group['attribute'], group['value']))

        # 'must_match' check still acts as a final validation step
        is_valid = True
        for attr in weights['must_match']:
            if rule_attrs.get(attr) != policy_flat.get(attr):
                logger.warning(
                    f"  -> Candidate '{rule_id}' failed final 'must_match' validation on '{attr}'. Discarding.")
                is_valid = False
                break
        if not is_valid:
            continue

        total_score, total_possible_weight, match_attributes = 0.0, 0.0, []

        for attr, weight in weights['should_match'].items():
            total_possible_weight += weight
            policy_value, rule_value = policy_flat.get(attr), rule_attrs.get(attr)

            if policy_value and rule_value:
                exact_score = 1.0 if policy_value == rule_value else 0.0
                jaccard_score = calculate_jaccard_similarity(policy_value, rule_value)
                cosine_score = \
                cosine_similarity([policy_embeddings.get(attr)], [embedding_model.encode(rule_value)])[0][0]

                attr_score = 0.0
                if exact_score == 1.0:
                    attr_score = 1.0
                elif cosine_score >= thresholds['cosine_similarity']:
                    attr_score = cosine_score
                elif jaccard_score >= thresholds['jaccard_similarity']:
                    attr_score = jaccard_score

                weighted_score = attr_score * weight
                total_score += weighted_score

                match_attributes.append({
                    "attribute_name": attr, "policy_value": policy_value, "rule_value": rule_value,
                    "jaccard_score": round(jaccard_score, 4), "cosine_score": round(float(cosine_score), 4),
                    "exact_match_score": exact_score, "composite_score": round(weighted_score, 4)
                })

        final_score = total_score / total_possible_weight if total_possible_weight > 0 else 0.0
        logger.info(f"  -> Calculated final score: {final_score:.4f}")

        final_results[rule_id] = {
            "overall_composite_score": round(final_score, 4),
            "attributes": match_attributes
        }

    # Sort final results by score
    sorted_items = sorted(final_results.items(), key=lambda item: item[1]['overall_composite_score'], reverse=True)
    return {k: v for k, v in sorted_items}


def main():
    """Main execution function to run the entire analysis pipeline."""
    logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(levelname)s - %(message)s')

    logger.info("🚀 Starting Policy Analyzer Script (Inverted Index Method)...")

    logger.info("\n--- Stage 1: Loading Data, Configs & Models ---")
    config = load_config()
    if not config: return

    try:
        rules_df = pd.read_csv(config['paths']['rules_csv'])
        with open(config['paths']['policy_json'], 'r', encoding='utf-8') as f:
            policy_data = json.load(f)

        policy_flat = flatten_json(policy_data)
        logger.info(f"✅ Policy and rules data loaded. Found {len(rules_df.id.unique())} unique rules.")
    except FileNotFoundError as e:
        logger.error(f"❌ Data file not found -> {e}");
        return

    # Build the inverted index (one-time setup)
    inverted_index = build_inverted_index(rules_df)

    logger.info(f"🤖 Loading embedding model ('{config['models']['embedding_model']}')...")
    model = SentenceTransformer(config['models']['embedding_model'])
    logger.info("🤖 Model loaded successfully.")

    # Get a small list of candidates using the index
    top_n = config.get('thresholds', {}).get('top_n_candidates', 10)
    candidate_ids = get_candidates_from_index(policy_flat, inverted_index, top_n)

    # Run detailed ranking only on the candidates
    if candidate_ids:
        results = rank_candidates(policy_flat, rules_df, candidate_ids, config, model)
    else:
        results = {}

    # Save and display results
    output_path = config['paths']['output_json']
    with open(output_path, 'w') as f:
        json.dump(results, f, indent=2)

    logger.info(f"\n🏁 Analysis Complete. Results saved to '{output_path}'")
    print("\n--- Final Output JSON ---\n")
    print(json.dumps(results, indent=2))


if __name__ == "__main__":
    main()