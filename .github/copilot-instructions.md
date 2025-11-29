<!-- Copied into repository by AI assistant. Keep concise and actionable. -->
# Copilot / AI Assistant Instructions for Trend-Analysis

This file contains focused, actionable guidance to help an AI coding agent be productive in this repository.

1. Project purpose (big picture)
- This repo provides a lightweight ML pipeline and a Streamlit app to estimate "viral potential" for short-form videos (TikTok/YouTube Shorts). Key flows: feature extraction -> model scoring -> explanation/recommendations.
- Primary runtime: `app.py` (Streamlit UI). Training and analysis live in `notebooks/02_Modeling.ipynb`.

2. Key files and responsibilities
- `app.py`: Streamlit front-end, loads model from `models/trend_model.pkl` and `models/feature_list.pkl`. If no model is present, uses a heuristic fallback. Example: saving a model in notebooks should use `joblib.dump(model, 'models/trend_model.pkl')`.
- `src/content_features.py`: Lightweight, pre-upload text features (e.g. `keyword_score`, `emotion_word_count`, `text_richness`). Use these functions to derive features before scoring.
- `src/model_utils.py`: Model scoring wrapper. `rate_video(metadata, model)` expects a numeric-only `metadata` dict whose keys match the model's training feature list.
- `src/explain.py`: Explanation helpers. `compute_feature_importance`, `rank_top_features`, and `generate_recommendations` contain the project-specific heuristics for explanations and textual recommendations.

3. Important patterns & conventions (project-specific)
- Numeric-only model rows: `app.py` builds a `metadata` dict with only numeric values to match `feature_list.pkl`. Maintain this shape when producing test rows.
- Feature list alignment: models that expose `feature_importances_` are used by `src/explain.py`; the code expects a `feature_list` (a list of feature names saved to `models/feature_list.pkl`) and indexes importances by position.
- Heuristic fallback: `app.py` contains a deterministic heuristic used when model file is missing; tests or local runs should account for both paths.
- Text feature functions are intentionally simple and deterministic—modify keyword sets in `src/content_features.py` rather than changing downstream code.

4. Developer workflows & common commands
- Run the demo UI locally: `streamlit run app.py` (from repo root). Ensure `requirements.txt` dependencies are installed.
- Train & save model: open `notebooks/02_Modeling.ipynb` and after training call `joblib.dump(model, 'models/trend_model.pkl')` and save the ordered `feature_list` via `joblib.dump(feature_list, 'models/feature_list.pkl')` so `app.py` can load them.
- If model load fails, `app.py` prints a Streamlit warning; fix by re-saving `models/trend_model.pkl` or ensuring `joblib` is available.

5. Integration points & external deps
- Optional sentiment support uses NLTK VADER in `app.py`. The code attempts to download `vader_lexicon` at runtime if missing. Keep this optional to avoid blocking UI when offline.
- Model artifacts: the app expects files under `models/` (not present in repo). CI or local dev should include a lightweight test model for validation.
- Data CSVs for EDA and modeling live in `data/` (e.g. `monthly_trends_2025.csv`, `youtube_shorts_tiktok_trends_2025.csv`). Use these in notebooks for reproducible training examples.

6. Example snippets the agent can reuse
- Build a numeric row to score (from `app.py`):
  - keys include `title_len`, `text_richness`, `like_rate`, `comment_rate`, `share_rate`, `views_per_day`, `likes_per_day`, `rel_like`, `rel_share`, `rel_combo`, `platform_cat`, `region_cat`, `language_cat`, `category_cat`, `creator_tier_cat`, etc.
- Call the scorer:
  ```py
  from src.model_utils import rate_video
  predicted_class, trend_score = rate_video(metadata, model)
  ```

7. Testing & debugging hints
- Unit-test targets: `src/content_features.py` (text parsing), `src/model_utils.py` (scoring interface), and `src/explain.py` (recommendation heuristics). Provide small fixtures that mimic `feature_list` ordering.
- When debugging model-loading issues, reproduce `load_model()` logic from `app.py` in a small script and verify `joblib.load` succeeds and `model.classes_` and `predict_proba` behave as expected.

8. What not to change without cross-checking
- Do not change the numeric feature names used in `app.py` unless you also update the training notebook and saved `feature_list.pkl`—the mapping is positional.
- Avoid making `src/content_features.py` asynchronous or I/O-bound; it is intentionally lightweight to run synchronously in the UI.

If any piece of this file looks unclear or you want more examples (unit-test skeletons, a small CI job to validate model load), tell me which section to expand. 
