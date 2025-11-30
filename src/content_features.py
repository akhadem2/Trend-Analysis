import re
import math

# Replace with your actual top trending hashtags derived in the notebook
import json, os

# Keyword/emotion seeds (fallbacks)
KEYWORDS_TREND = {"challenge", "dance", "tutorial", "hack", "giveaway", "ai", "filter"}
EMOTION_WORDS = {"amazing", "crazy", "fun", "wild", "insane", "emotional", "cute", "wow"}

DEFAULT_TOP_TAGS = {"#dance", "#challenge", "#viral", "#trend", "#fyp", "#tiktok", "#shorts", "#ai"}
tags_path = os.path.join("models", "top_trending_hashtags.json")
try:
    TOP_TRENDING_HASHTAGS = set(json.load(open(tags_path))) if os.path.exists(tags_path) else DEFAULT_TOP_TAGS
except Exception:
    TOP_TRENDING_HASHTAGS = DEFAULT_TOP_TAGS

    
def extract_content_features(title: str, description: str):
    """Derive simple text-based features from a title and description.

    These features are intentionally lightweight so they can be estimated
    before upload. Adjust keyword lists above for domain tuning.
    """
    txt = f"{title} {description}".lower()
    tokens = re.findall(r"[a-z0-9#]+", txt)
    length = len(tokens)
    unique = len(set(tokens))
    richness = unique / (length + 1e-6)

    keyword_hits = sum(1 for t in tokens if t in KEYWORDS_TREND)
    emotion_hits = sum(1 for t in tokens if t in EMOTION_WORDS)
    hashtags = sum(1 for t in tokens if t.startswith("#"))
    trending_hashtag_hits = sum(1 for t in tokens if t in TOP_TRENDING_HASHTAGS)
    trending_hashtag_ratio = trending_hashtag_hits / (hashtags + 1e-6)

    return {
        "title_len": len(title),
        "text_richness": richness,
        "keyword_score": keyword_hits,
        "emotion_word_count": emotion_hits,
        "hashtag_count": hashtags,
        "log_token_len": math.log1p(length),
        "trending_hashtag_hits": trending_hashtag_hits,
        "trending_hashtag_ratio": trending_hashtag_ratio,
    }
