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

def load_label_map(col, cat_col, csv_path="data/youtube_shorts_tiktok_trends_2025.csv_ML.csv"):
    try:
        df_map = pd.read_csv(csv_path, usecols=[col, cat_col]).dropna().drop_duplicates()
        df_map = df_map.sort_values(cat_col)
        return [(row[col], int(row[cat_col])) for _, row in df_map.iterrows()]
    except Exception as e:
        st.warning(f"Could not load {col} mapping: {e}")
        return []

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
label_encoder_path = os.path.join("models", "label_encoder.pkl")
if model is not None and os.path.exists(label_encoder_path) and joblib is not None:
    try:
        model.label_encoder_ = joblib.load(label_encoder_path)
        model.class_labels_ = model.label_encoder_.classes_
    except Exception as e:
        st.warning(f"Could not load label encoder: {e}")

feature_list = None
feature_list_path = os.path.join("models", "feature_list.pkl")
if os.path.exists(feature_list_path) and joblib is not None:
    try:
        feature_list = joblib.load(feature_list_path)
    except Exception as e:
        st.warning(f"Could not load feature list: {e}")

# Load metrics
metrics_path = os.path.join("models", "metrics.json")
metrics_data = None
if os.path.exists(metrics_path):
    try:
        import json
        with open(metrics_path) as f:
            metrics_data = json.load(f)
    except Exception as e:
        st.warning(f"Could not load metrics: {e}")

# Display Model Performance Dashboard
if metrics_data or feature_list:
    st.header("📊 Model Performance Dashboard")
    
    # Model Comparison
    if metrics_data:
        st.subheader("Model Comparison")
        
        model_names = []
        cv_f1_means = []
        cv_f1_stds = []
        holdout_accs = []
        holdout_f1s = []
        
        for model_name, metrics in metrics_data.items():
            if model_name != "selected" and isinstance(metrics, dict):
                display_name = model_name.replace("_", " ").title()
                model_names.append(display_name)
                cv_f1_means.append(metrics.get("cv_f1_mean", 0))
                cv_f1_stds.append(metrics.get("cv_f1_std", 0))
                holdout_accs.append(metrics.get("acc", 0))
                holdout_f1s.append(metrics.get("f1_macro", 0))
        
        col1, col2 = st.columns(2)
        
        with col1:
            st.markdown("**Cross-Validation F1 Scores**")
            cv_chart_data = pd.DataFrame({
                "Model": model_names,
                "CV F1 Score": cv_f1_means
            })
            st.bar_chart(cv_chart_data.set_index("Model"))
        
        with col2:
            st.markdown("**Holdout Test Accuracy**")
            holdout_chart_data = pd.DataFrame({
                "Model": model_names,
                "Accuracy": holdout_accs
            })
            st.bar_chart(holdout_chart_data.set_index("Model"))
        
        # Detailed metrics table
        st.markdown("**Detailed Metrics**")
        metrics_df = pd.DataFrame({
            "Model": model_names,
            "CV F1 (mean)": [f"{x:.4f}" for x in cv_f1_means],
            "CV F1 (std)": [f"{x:.4f}" for x in cv_f1_stds],
            "Test Accuracy": [f"{x:.4f}" for x in holdout_accs],
            "Test F1-Macro": [f"{x:.4f}" for x in holdout_f1s]
        })
        st.dataframe(metrics_df, use_container_width=True, hide_index=True)
        
        selected_model = metrics_data.get("selected", "unknown")
        st.success(f"✓ Selected Model: **{selected_model.replace('_', ' ').title()}**")
    
    # Feature Importance
    if feature_list and model is not None:
        st.subheader("Top Feature Importance")
        try:
            importance_map = compute_feature_importance(model, feature_list)
            top_feats = rank_top_features(importance_map, top_n=10)
            
            importance_df = pd.DataFrame({
                "Feature": [f["feature"] for f in top_feats],
                "Importance": [f["importance"] for f in top_feats]
            })
            
            col1, col2 = st.columns([2, 1])
            with col1:
                st.bar_chart(importance_df.set_index("Feature"))
            with col2:
                st.markdown("**Ranked Features**")
                for i, f in enumerate(top_feats, 1):
                    st.text(f"{i}. {f['feature']}: {f['importance']:.4f}")
        except Exception as e:
            st.warning(f"Could not compute feature importance: {e}")
    
    st.divider()

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
    platform_options = load_label_map("platform", "platform_cat")
    region_options = load_label_map("region", "region_cat")
    language_options = load_label_map("language", "language_cat")
    category_options = load_label_map("category", "category_cat")
    creator_tier_options = load_label_map("creator_tier", "creator_tier_cat")

    platform_sel = st.selectbox("Platform", platform_options, format_func=lambda x: x[0])
    platform_cat = platform_sel[1] if platform_sel else 2  # fallback

    region_sel = st.selectbox("Region", region_options, format_func=lambda x: x[0])
    region_cat = region_sel[1] if region_sel else 5

    language_sel = st.selectbox("Language", language_options, format_func=lambda x: x[0])
    language_cat = language_sel[1] if language_sel else 1

    category_sel = st.selectbox("Category", category_options, format_func=lambda x: x[0])
    category_cat = category_sel[1] if category_sel else 7

    creator_tier_sel = st.selectbox("Creator tier", creator_tier_options, format_func=lambda x: x[0])
    creator_tier_cat = creator_tier_sel[1] if creator_tier_sel else 1

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
        "title_sentiment": sentiment_score or 0.0,
        "trending_hashtag_hits": content_feats.get("trending_hashtag_hits", 0),
        "trending_hashtag_ratio": content_feats.get("trending_hashtag_ratio", 0.0),

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


    st.subheader("Model Snapshot")
if model is None:
    st.info("No trained model found; using heuristic fallback.")
else:
    # Load label encoder if you saved it

    metrics_path = os.path.join("models", "metrics.json")
    if os.path.exists(metrics_path):
        import json
        with open(metrics_path) as f:
            m = json.load(f)

        # Show metrics for the selected model if present, otherwise fallback
        selected_key = m.get("selected")
        if selected_key and selected_key in m and isinstance(m[selected_key], dict):
            sel = m[selected_key]
            cv_mean = sel.get("cv_f1_mean")
            cv_std = sel.get("cv_f1_std")
            acc = sel.get("acc")
            f1_macro = sel.get("f1_macro")
            label = selected_key
        else:
            # Legacy single-model shape: { "cv": {...}, "holdout": {...} }
            sel = None
            cv = m.get("cv", {})
            holdout = m.get("holdout", {})
            cv_mean = cv.get("f1_mean")
            cv_std = cv.get("f1_std")
            acc = holdout.get("accuracy")
            f1_macro = holdout.get("f1_macro")
            label = "model"




    with st.expander("Inputs expected by the model"):
        st.json(feature_list or ["No feature_list.pkl loaded"])


st.caption("Disclaimer: This is an estimation tool; real performance depends on platform dynamics, timing, and audience.")
