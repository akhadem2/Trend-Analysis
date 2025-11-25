import re
import math

KEYWORDS_TREND = {"challenge", "dance", "tutorial", "hack", "giveaway", "ai", "filter"}
EMOTION_WORDS = {"amazing", "crazy", "fun", "wild", "insane", "emotional", "cute", "wow"}

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

    return {
        "title_len": len(title),
        "text_richness": richness,
        "keyword_score": keyword_hits,
        "emotion_word_count": emotion_hits,
        "hashtag_count": hashtags,
        "log_token_len": math.log1p(length)
    }
