import pandas as pd

def rate_video(metadata, model):
    row = pd.DataFrame([metadata])
    proba = model.predict_proba(row)[0]

    if hasattr(model, "label_encoder_"):
        pred_int = model.predict(row)[0]
        predicted_class = model.label_encoder_.inverse_transform([pred_int])[0]
        classes = list(model.label_encoder_.classes_)
    else:
        classes = list(getattr(model, "class_labels_", None) or getattr(model, "classes_", []))
        predicted_class = model.predict(row)[0]

    trend_score = None
    if "trending" in classes:
        trend_score = proba[classes.index("trending")] * 100

    return predicted_class, trend_score



