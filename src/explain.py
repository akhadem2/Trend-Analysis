import math
from typing import List, Dict, Any

def compute_feature_importance(model, feature_list: List[str]) -> Dict[str, float]:
    """Return a mapping of feature -> importance if model supports it."""
    if hasattr(model, 'feature_importances_'):
        importances = model.feature_importances_
        return {f: float(importances[i]) for i, f in enumerate(feature_list) if i < len(importances)}
    return {f: 0.0 for f in feature_list}

def rank_top_features(importance_map: Dict[str, float], top_n: int = 8) -> List[Dict[str, Any]]:
    items = sorted(importance_map.items(), key=lambda x: x[1], reverse=True)
    return [{"feature": k, "importance": v} for k, v in items[:top_n]]

def generate_recommendations(metadata: Dict[str, Any], importance_map: Dict[str, float]) -> List[str]:
    recs = []
    # Heuristic thresholds
    like_rate = metadata.get('like_rate', 0)
    share_rate = metadata.get('share_rate', 0)
    hashtag_count = metadata.get('hashtag_count', metadata.get('title_len', 0))
    text_richness = metadata.get('text_richness', 0)

    # Engagement suggestions
    if like_rate < 0.05:
        recs.append("Increase early like rate: add a stronger hook in first 2 seconds.")
    if share_rate < 0.008:
        recs.append("Encourage sharing: pose a challenge or call-to-action in caption.")
    if hashtag_count < 3:
        recs.append("Add a couple relevant, niche hashtags to improve discovery.")
    if text_richness < 0.4:
        recs.append("Caption is low in unique tokens; consider adding specific keywords or context.")

    # Importance-driven hints
    top_eng_features = [f for f, imp in sorted(importance_map.items(), key=lambda x: x[1], reverse=True) if 'rate' in f][:3]
    if top_eng_features:
        recs.append(f"Focus on improving: {', '.join(top_eng_features)} — they drive trend classification.")

    if not recs:
        recs.append("Content already shows strong signals; experiment with timing and thumbnail selection.")
    return recs

def format_importance_table(top_features: List[Dict[str, Any]]) -> str:
    lines = ["Feature | Importance", "------------------"]
    for item in top_features:
        lines.append(f"{item['feature']} | {item['importance']:.4f}")
    return "\n".join(lines)
