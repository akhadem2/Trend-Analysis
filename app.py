import os
import pandas as pd
import streamlit as st

from src.model_utils import rate_video
from src.content_features import extract_content_features
from src.explain import compute_feature_importance, rank_top_features, generate_recommendations, format_importance_table

# Optional sentiment (VADER) - graceful fallback
try:
    from nltk.sentiment import SentimentIntensityAnalyzer
    import nltk
    try:
        nltk.data.find('sentiment/vader_lexicon.zip')
    except LookupError:
        nltk.download('vader_lexicon')
    _sentiment = SentimentIntensityAnalyzer()
except Exception:
    _sentiment = None

try:
    import joblib
except ImportError:  # joblib is bundled with sklearn, fallback
    joblib = None


MODEL_PATH = os.path.join("models", "trend_model.pkl")

st.set_page_config(page_title="Short Form Video Trend Scorer", layout="centered")
st.title("📈 Viral Potential Estimator")
st.caption("Estimate trending probability for a planned TikTok/Short before uploading.")

with st.expander("How this works"):
    st.write(
        """This tool loads a trained classifier (you train & save it in the notebooks) and
        derives simple metadata + lightweight text features from your planned video. It then
        produces a class prediction and a probability score for the 'trending' class (if present).
        If no trained model file is found, a placeholder estimation logic is used so you can still
        experiment with inputs.
        """
    )


def load_model():
    if os.path.exists(MODEL_PATH) and joblib is not None:
        try:
            return joblib.load(MODEL_PATH)
        except Exception as e:
            st.warning(f"Failed to load model: {e}")
    return None


model = load_model()
# Load feature list if present for importance ranking
feature_list = None
feature_list_path = os.path.join("models", "feature_list.pkl")
if os.path.exists(feature_list_path) and joblib is not None:
    try:
        feature_list = joblib.load(feature_list_path)
    except Exception as e:
        st.warning(f"Could not load feature list: {e}")
if model is None:
    st.warning(
        "No trained model found at models/trend_model.pkl. Train in 02_Modeling.ipynb and save via joblib.dump(model, 'models/trend_model.pkl'). Using heuristic fallback.")

st.subheader("Enter Planned Video Info")

with st.form("video_form"):
    title = st.text_input("Video Title", "Crazy dance challenge with new AI filter")
    description = st.text_area("Description / Caption", "Try this insane move #dance #challenge #ai")
    video_length = st.number_input("Length (seconds)", min_value=1, max_value=600, value=20)

    st.markdown("### Estimated First 7 Days Metrics (your expectation)")
    views_7d = st.number_input("Expected Views (7d)", min_value=0, value=50000, step=1000)
    likes_7d = st.number_input("Expected Likes (7d)", min_value=0, value=4000, step=100)
    comments_7d = st.number_input("Expected Comments (7d)", min_value=0, value=450, step=10)
    shares_7d = st.number_input("Expected Shares (7d)", min_value=0, value=300, step=10)

    st.markdown("### Context (optional rough categories)")
    trend_label = st.selectbox("Prior performance label (if any)", ["rising", "seasonal", "stable", "declining"], index=0)
    category_cat = st.number_input("Category code", min_value=0, value=7)
    region_cat = st.number_input("Region code", min_value=0, value=5)
    language_cat = st.number_input("Language code", min_value=0, value=1)
    platform_cat = st.number_input("Platform code", min_value=0, value=2)
    creator_tier_cat = st.number_input("Creator tier code", min_value=0, value=1)

    submitted = st.form_submit_button("Estimate Viral Potential")

if submitted:
    # Derived engagement rates (avoid division by zero)
    like_rate = likes_7d / views_7d if views_7d else 0
    comment_rate = comments_7d / views_7d if views_7d else 0
    share_rate = shares_7d / views_7d if views_7d else 0
    
    # Compute derived metrics
    views_per_day = views_7d / 7 if views_7d else 0
    likes_per_day = likes_7d / 7 if likes_7d else 0
    
    # More realistic baselines for viral content (stricter thresholds)
    viral_like_baseline = 0.10  # 10% like rate for viral
    viral_share_baseline = 0.015  # 1.5% share rate for viral
    
    # Relative metrics capped at 2.0 to prevent inflation
    rel_like = min(2.0, like_rate / viral_like_baseline) if like_rate else 0
    rel_share = min(2.0, share_rate / viral_share_baseline) if share_rate else 0
    rel_combo = (rel_like + rel_share) / 2
    
    # Penalty for low absolute engagement (below 10k views is penalized)
    engagement_penalty = min(1.0, views_7d / 10000) if views_7d else 0.1

    content_feats = extract_content_features(title, description)

    # Sentiment (not used in prediction, only explanation)
    sentiment_score = None
    if _sentiment is not None:
        sentiment_score = _sentiment.polarity_scores(f"{title} {description}")['compound']

    # Build metadata matching exact training features (numeric only)
    metadata = {
        "title_len": content_feats["title_len"],
        "text_richness": content_feats["text_richness"],
        "like_rate": like_rate * engagement_penalty,
        "comment_rate": comment_rate * engagement_penalty,
        "share_rate": share_rate * engagement_penalty,
        "views_per_day": views_per_day,
        "likes_per_day": likes_per_day,
        "rel_like": rel_like * engagement_penalty,
        "rel_share": rel_share * engagement_penalty,
        "rel_combo": rel_combo * engagement_penalty,
        "like_hashtag_interaction": like_rate * content_feats["hashtag_count"] * engagement_penalty,
        "share_hashtag_interaction": share_rate * content_feats["hashtag_count"] * engagement_penalty,
        "platform_cat": platform_cat,
        "region_cat": region_cat,
        "language_cat": language_cat,
        "category_cat": category_cat,
        "traffic_source_cat": 3,  # default
        "device_brand_cat": 4,  # default
        "creator_tier_cat": creator_tier_cat,
        "richness_traffic_interaction": content_feats["text_richness"] * 3,
        "weekend_hashtag_boost": 0,  # default
        "region_platform_avg_views_per_day": max(50000, views_per_day * 1.5),  # assume platform avg is higher
        "region_platform_avg_like_rate": viral_like_baseline,
        "region_platform_avg_share_rate": viral_share_baseline,
        "region_platform_avg_rel_like": 1.0,
        "region_platform_avg_rel_share": 1.0,
    }

    if model is not None:
        predicted_class, trend_score = rate_video(metadata, model)
        st.success(f"Predicted bucket: {predicted_class}")
        if trend_score is not None:
            st.info(f"Trending probability: {trend_score:.2f}%")
        else:
            st.info("Model has no 'trending' class; probability unavailable.")

        # Explanations section
        with st.expander("Why this prediction?"):
            if feature_list is not None:
                importance_map = compute_feature_importance(model, feature_list)
                top_feats = rank_top_features(importance_map)
                st.markdown("**Top Driving Features**")
                st.code(format_importance_table(top_feats))
                # Merge metadata + derived content for recommendations
                meta_for_recs = {**metadata, **content_feats}
                recs = generate_recommendations(meta_for_recs, importance_map)
                st.markdown("**Recommendations to Improve**")
                for r in recs:
                    st.write("- " + r)
            else:
                st.write("Feature list not found; re-run notebook cell to save 'feature_list.pkl'.")
            if sentiment_score is not None:
                st.markdown(f"**Caption Sentiment (VADER compound):** {sentiment_score:.3f}")
                if sentiment_score < 0:
                    st.write("Consider a more positive or curiosity-driven phrasing to boost engagement.")
                elif sentiment_score > 0.7:
                    st.write("High positive sentiment; balance with a hook or question to drive interaction.")
    else:
        # Simple heuristic fallback: combine signals
        heuristic = (
            (like_rate * 100) * 0.5
            + (share_rate * 100) * 0.3
            + content_feats["keyword_score"] * 2
            + content_feats["emotion_word_count"] * 1.5
        )
        trending_prob = max(0, min(100, heuristic))
        bucket = "trending" if trending_prob > 60 else ("likely" if trending_prob > 35 else "low")
        st.success(f"(Heuristic) Estimated bucket: {bucket}")
        st.info(f"(Heuristic) Trending probability: {trending_prob:.2f}%")

    with st.expander("Raw numeric features"):
        st.json(metadata)
    with st.expander("Derived content features"):
        st.json(content_feats)

st.caption("Disclaimer: This is an estimation tool; real performance depends on platform dynamics, timing, and audience.")
