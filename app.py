"""Streamlit app for the packaged caddisfly image classifier."""

from __future__ import annotations

import base64
import io
import traceback
from html import escape
from typing import Any, Dict, List, Tuple

import streamlit as st

from inference import (
    InferenceError,
    combine_probabilities,
    load_image_from_bytes,
    load_model_artifacts,
    predict_image_bytes,
    summarize_probabilities,
)


CONFIDENCE_THRESHOLD = 0.70
SUPPORTED_UPLOAD_TYPES = ("jpg", "jpeg", "png", "webp")
SESSION_KEYS = (
    "first_image_bytes",
    "first_image_name",
    "first_prediction",
    "second_image_bytes",
    "second_image_name",
    "second_prediction",
    "combined_prediction",
)


st.set_page_config(
    page_title="Caddisfly Image Classifier",
    page_icon="C",
    layout="wide",
)


def apply_page_style() -> None:
    st.markdown(
        """
        <style>
        .block-container {
            max-width: 1250px;
            padding-top: 2rem;
            padding-bottom: 3rem;
        }
        h1, h2, h3, p {
            letter-spacing: 0;
        }
        .app-title {
            color: #17324D;
            font-size: 2rem;
            font-weight: 760;
            line-height: 1.15;
            margin: 0 0 0.35rem;
        }
        .app-subtitle {
            color: #475569;
            font-size: 1rem;
            line-height: 1.45;
            margin: 0 0 0.75rem;
        }
        .result-card {
            border: 1px solid #DCE3E8;
            border-radius: 10px;
            padding: 1rem 1.1rem 1.05rem;
            background: #ffffff;
            margin: 0 0 0.75rem;
            box-shadow: 0 1px 2px rgba(15, 23, 42, 0.04);
        }
        .result-label {
            color: #2A7F8E;
            font-size: 0.78rem;
            font-weight: 700;
            text-transform: uppercase;
            letter-spacing: 0.04em;
            margin-bottom: 0.25rem;
        }
        .result-title {
            color: #17324D;
            font-size: 1.22rem;
            font-weight: 760;
            line-height: 1.35;
            margin-bottom: 0.7rem;
            overflow-wrap: anywhere;
        }
        .field-grid {
            display: grid;
            grid-template-columns: repeat(3, minmax(0, 1fr));
            gap: 0.45rem;
            margin: 0.75rem 0;
        }
        .field-box {
            background: #F4F7F9;
            border: 1px solid #DCE3E8;
            border-radius: 8px;
            padding: 0.55rem 0.65rem;
        }
        .field-name {
            color: #64748b;
            font-size: 0.72rem;
            font-weight: 700;
            text-transform: uppercase;
        }
        .field-value {
            color: #1e293b;
            font-size: 0.92rem;
            margin-top: 0.1rem;
            overflow-wrap: anywhere;
        }
        .confidence-line {
            color: #17324D;
            font-size: 1rem;
            margin-top: 0.35rem;
        }
        .status-pill {
            display: inline-block;
            border-radius: 999px;
            font-size: 0.85rem;
            font-weight: 700;
            padding: 0.25rem 0.6rem;
            margin-top: 0.55rem;
        }
        .status-good {
            background: rgba(46, 125, 91, 0.12);
            color: #2E7D5B;
        }
        .status-warn {
            background: rgba(183, 121, 31, 0.14);
            color: #B7791F;
        }
        .interpretation {
            color: #475569;
            font-size: 0.92rem;
            line-height: 1.45;
            margin-top: 0.6rem;
        }
        .image-frame {
            border: 1px solid #DCE3E8;
            border-radius: 10px;
            background: #F4F7F9;
            padding: 0.65rem;
            margin-bottom: 0.75rem;
            text-align: center;
        }
        .image-frame img {
            max-height: 460px;
            width: 100%;
            object-fit: contain;
            border-radius: 8px;
            display: block;
            margin: 0 auto;
        }
        .image-frame.compact img {
            max-height: 230px;
        }
        .image-caption {
            color: #64748b;
            font-size: 0.82rem;
            margin-top: 0.45rem;
            overflow-wrap: anywhere;
        }
        .top5-list {
            display: grid;
            gap: 0.72rem;
            margin: 0.2rem 0 0.35rem;
        }
        .top5-row {
            border: 1px solid #DCE3E8;
            border-radius: 8px;
            background: #ffffff;
            padding: 0.65rem 0.75rem;
        }
        .top5-meta {
            display: grid;
            grid-template-columns: 2rem minmax(0, 1fr) 5.5rem;
            gap: 0.55rem;
            align-items: start;
            color: #17324D;
            font-size: 0.9rem;
        }
        .top5-rank {
            color: #2A7F8E;
            font-weight: 760;
        }
        .top5-name {
            overflow-wrap: anywhere;
            line-height: 1.35;
        }
        .top5-prob {
            color: #17324D;
            font-weight: 760;
            text-align: right;
        }
        .prob-track {
            height: 0.42rem;
            border-radius: 999px;
            background: #E5ECF1;
            overflow: hidden;
            margin-top: 0.45rem;
        }
        .prob-fill {
            height: 100%;
            border-radius: 999px;
            background: #2A7F8E;
        }
        .compact-heading {
            color: #17324D;
            font-size: 1.25rem;
            font-weight: 740;
            margin: 0.5rem 0 0.15rem;
        }
        .helper-text {
            color: #475569;
            font-size: 0.94rem;
            line-height: 1.45;
            margin-bottom: 0.7rem;
        }
        .mini-result {
            border: 1px solid #DCE3E8;
            border-radius: 9px;
            background: #ffffff;
            padding: 0.75rem 0.85rem;
            margin-bottom: 0.75rem;
        }
        .mini-title {
            color: #64748b;
            font-size: 0.78rem;
            font-weight: 700;
            text-transform: uppercase;
            margin-bottom: 0.25rem;
        }
        .mini-label {
            color: #17324D;
            font-weight: 720;
            line-height: 1.35;
            overflow-wrap: anywhere;
        }
        .mini-confidence {
            color: #475569;
            margin-top: 0.35rem;
            font-size: 0.92rem;
        }
        .footer-note {
            color: #64748b;
            border-top: 1px solid #DCE3E8;
            font-size: 0.86rem;
            line-height: 1.45;
            margin-top: 1.25rem;
            padding-top: 0.8rem;
        }
        @media (max-width: 760px) {
            .field-grid { grid-template-columns: 1fr; }
            .top5-meta {
                grid-template-columns: 1.6rem minmax(0, 1fr);
            }
            .top5-prob {
                grid-column: 2;
                text-align: left;
            }
            .app-title {
                font-size: 1.7rem;
            }
        }
        </style>
        """,
        unsafe_allow_html=True,
    )


@st.cache_resource(show_spinner="Loading trained MobileNetV2 model...")
def get_model_artifacts():
    return load_model_artifacts(device_name="cpu")


def initialise_session_state() -> None:
    if "uploader_nonce" not in st.session_state:
        st.session_state.uploader_nonce = 0
    for key in SESSION_KEYS:
        st.session_state.setdefault(key, None)


def reset_application() -> None:
    next_nonce = int(st.session_state.get("uploader_nonce", 0)) + 1
    for key in list(st.session_state.keys()):
        if (
            key in SESSION_KEYS
            or key.startswith("first_uploader_")
            or key.startswith("second_uploader_")
        ):
            del st.session_state[key]
    st.session_state.uploader_nonce = next_nonce


def format_percent(value: float) -> str:
    return f"{value * 100.0:.2f}%"


def parse_class_label(label: str) -> Tuple[str, str, str]:
    sex = "Not parsed"
    core = label
    for candidate in (" female", " male"):
        if label.endswith(candidate):
            sex = candidate.strip()
            core = label[: -len(candidate)].strip()
            break

    parts = core.split(maxsplit=1)
    if not parts:
        return "Not parsed", "Not parsed", sex
    family = parts[0].rstrip(",")
    taxon = parts[1] if len(parts) > 1 else "Not parsed"
    return family, taxon, sex


def show_top5(top_predictions: List[Dict[str, Any]], title: str, expanded: bool = True) -> None:
    html_parts = ['<div class="top5-list">']
    for item in top_predictions:
        probability = float(item["probability"])
        width = max(0.0, min(probability * 100.0, 100.0))
        html_parts.append(
            '<div class="top5-row">'
            '<div class="top5-meta">'
            f'<div class="top5-rank">{int(item["rank"])}</div>'
            f'<div class="top5-name">{str(item["class_name"])}</div>'
            f'<div class="top5-prob">{format_percent(probability)}</div>'
            '</div>'
            '<div class="prob-track">'
            f'<div class="prob-fill" style="width: {width:.2f}%"></div>'
            '</div>'
            '</div>'
        )
    html_parts.append("</div>")
    html_content = "".join(html_parts)

    with st.expander(title, expanded=expanded):
        st.markdown(html_content, unsafe_allow_html=True)


def confidence_status(confidence: float, threshold: float) -> Tuple[str, str]:
    if confidence >= threshold:
        return "High-confidence prediction", "status-good"
    return "Uncertain prediction", "status-warn"


def result_card(
    prediction: Dict[str, Any],
    threshold: float,
    heading: str = "Prediction",
    note: str | None = None,
) -> None:
    confidence = float(prediction["confidence"])
    prediction_label = str(prediction["prediction"])
    family, taxon, sex = parse_class_label(prediction_label)
    status, status_class = confidence_status(confidence, threshold)

    st.markdown(
        f"""
        <div class="result-card">
            <div class="result-label">{escape(heading)}</div>
            <div class="result-title">{escape(prediction_label)}</div>
            <div class="field-grid">
                <div class="field-box">
                    <div class="field-name">Family</div>
                    <div class="field-value">{escape(family)}</div>
                </div>
                <div class="field-box">
                    <div class="field-name">Taxon</div>
                    <div class="field-value">{escape(taxon)}</div>
                </div>
                <div class="field-box">
                    <div class="field-name">Sex</div>
                    <div class="field-value">{escape(sex)}</div>
                </div>
            </div>
            <div class="confidence-line">Confidence: <strong>{format_percent(confidence)}</strong></div>
            <div class="status-pill {status_class}">{status}</div>
            {f'<div class="interpretation">{escape(note)}</div>' if note else ''}
        </div>
        """,
        unsafe_allow_html=True,
    )


def mini_result_card(title: str, prediction: Dict[str, Any]) -> None:
    confidence = float(prediction["confidence"])
    st.markdown(
        f"""
        <div class="mini-result">
            <div class="mini-title">{escape(title)}</div>
            <div class="mini-label">{escape(str(prediction["prediction"]))}</div>
            <div class="mini-confidence">Confidence: <strong>{format_percent(confidence)}</strong></div>
        </div>
        """,
        unsafe_allow_html=True,
    )


def image_data_url(image_bytes: bytes) -> str:
    image = load_image_from_bytes(image_bytes)
    buffer = io.BytesIO()
    image.save(buffer, format="PNG")
    encoded = base64.b64encode(buffer.getvalue()).decode("ascii")
    return f"data:image/png;base64,{encoded}"


def preview_image(image_bytes: bytes, caption: str, compact: bool = False) -> None:
    image = load_image_from_bytes(image_bytes)
    del image
    class_name = "image-frame compact" if compact else "image-frame"
    st.markdown(
        f"""
        <div class="{class_name}">
            <img src="{image_data_url(image_bytes)}" alt="{escape(caption)}">
            <div class="image-caption">{escape(caption)}</div>
        </div>
        """,
        unsafe_allow_html=True,
    )


def store_first_upload(uploaded_file) -> None:
    image_bytes = uploaded_file.getvalue()
    if image_bytes != st.session_state.first_image_bytes:
        st.session_state.first_image_bytes = image_bytes
        st.session_state.first_image_name = uploaded_file.name
        st.session_state.first_prediction = None
        st.session_state.second_image_bytes = None
        st.session_state.second_image_name = None
        st.session_state.second_prediction = None
        st.session_state.combined_prediction = None


def store_second_upload(uploaded_file) -> None:
    image_bytes = uploaded_file.getvalue()
    if image_bytes != st.session_state.second_image_bytes:
        st.session_state.second_image_bytes = image_bytes
        st.session_state.second_image_name = uploaded_file.name
        st.session_state.second_prediction = None
        st.session_state.combined_prediction = None


def show_error(message: str, exc: Exception) -> None:
    st.error(message)
    with st.expander("Technical details"):
        st.code("".join(traceback.format_exception(type(exc), exc, exc.__traceback__)))


def predict_if_needed(state_key: str, image_key: str, artifacts) -> None:
    if st.session_state[state_key] is None and st.session_state[image_key] is not None:
        st.session_state[state_key] = predict_image_bytes(
            st.session_state[image_key],
            artifacts=artifacts,
            top_k=5,
        )


def show_model_information(artifacts) -> None:
    with st.expander("Model information"):
        st.markdown(
            "\n".join(
                [
                    f"- Model architecture: {artifacts.config.get('model_name', 'MobileNetV2')}",
                    f"- Number of classes: {len(artifacts.class_names)}",
                    f"- Device: {artifacts.device}",
                    f"- Confidence threshold: {format_percent(CONFIDENCE_THRESHOLD)}",
                    f"- Supported image formats: {', '.join(file_type.upper() for file_type in SUPPORTED_UPLOAD_TYPES)}",
                    "- Two-image method: average the two 50-class softmax probability vectors class by class.",
                ]
            )
        )


def show_first_image_workflow(artifacts) -> None:
    first_file = st.file_uploader(
        "Upload the first insect image",
        type=SUPPORTED_UPLOAD_TYPES,
        key=f"first_uploader_{st.session_state.uploader_nonce}",
        help="Accepted formats: JPG, JPEG, PNG, and WEBP.",
    )
    if first_file is not None:
        store_first_upload(first_file)

    if st.session_state.first_image_bytes is None:
        st.info("Upload one clear image to begin.")
        return

    image_col, result_col = st.columns([1.05, 1], gap="large")
    with image_col:
        preview_image(st.session_state.first_image_bytes, st.session_state.first_image_name or "First image")

    with result_col:
        predict_if_needed("first_prediction", "first_image_bytes", artifacts)
        first_prediction = st.session_state.first_prediction
        confidence = float(first_prediction["confidence"])
        note = (
            "Treat this as the current identification for this model."
            if confidence >= CONFIDENCE_THRESHOLD
            else "The model is uncertain. Upload another image of the same specimen from a different angle."
        )
        result_card(first_prediction, CONFIDENCE_THRESHOLD, note=note)
        if float(first_prediction["confidence"]) >= CONFIDENCE_THRESHOLD:
            pass
        else:
            st.warning(
                "The model is uncertain. Upload another image of the same specimen from a different angle."
            )
        show_top5(first_prediction["top_k"], "View Top-5 predictions", expanded=True)


def show_second_image_workflow(artifacts) -> None:
    first_prediction = st.session_state.first_prediction
    if first_prediction is None or float(first_prediction["confidence"]) >= CONFIDENCE_THRESHOLD:
        return

    st.markdown('<div class="compact-heading">Add a second image</div>', unsafe_allow_html=True)
    st.markdown(
        """
        <div class="helper-text">
        Use another image of the same specimen with a different angle, clearer body features,
        better lighting, and less background obstruction.
        </div>
        """,
        unsafe_allow_html=True,
    )

    second_file = st.file_uploader(
        "Upload a second image of the same insect",
        type=SUPPORTED_UPLOAD_TYPES,
        key=f"second_uploader_{st.session_state.uploader_nonce}",
    )
    if second_file is not None:
        store_second_upload(second_file)

    if st.session_state.second_image_bytes is None:
        st.info("Upload one second image if available. The app will not request additional images.")
        return

    predict_if_needed("second_prediction", "second_image_bytes", artifacts)
    second_prediction = st.session_state.second_prediction

    result_cols = st.columns(2, gap="large")
    with result_cols[0]:
        st.markdown("**Image 1 result**")
        preview_image(st.session_state.first_image_bytes, st.session_state.first_image_name or "First image", compact=True)
        mini_result_card("Image 1", first_prediction)
        show_top5(first_prediction["top_k"], "View Image 1 Top-5 predictions", expanded=False)
    with result_cols[1]:
        st.markdown("**Image 2 result**")
        preview_image(st.session_state.second_image_bytes, st.session_state.second_image_name or "Second image", compact=True)
        mini_result_card("Image 2", second_prediction)
        show_top5(second_prediction["top_k"], "View Image 2 Top-5 predictions", expanded=False)

    combined_probabilities = combine_probabilities(
        first_prediction["probabilities"],
        st.session_state.second_prediction["probabilities"],
    )
    st.session_state.combined_prediction = summarize_probabilities(
        combined_probabilities,
        artifacts.class_names,
        k=5,
    )

    show_combined_result(first_prediction, st.session_state.second_prediction, st.session_state.combined_prediction)


def show_combined_result(
    first_prediction: Dict[str, Any],
    second_prediction: Dict[str, Any],
    combined_prediction: Dict[str, Any],
) -> None:
    first_label = first_prediction["prediction"]
    second_label = second_prediction["prediction"]
    combined_confidence = float(combined_prediction["confidence"])

    if first_label == second_label:
        agreement_note = "Both images agree."
    else:
        agreement_note = "The two images produced different predictions."

    if combined_confidence < CONFIDENCE_THRESHOLD:
        confidence_note = "The combined result is still uncertain. Please use a clearer image or consult an expert."
    elif first_label == second_label:
        confidence_note = "The shared class is used as the final prediction."
    else:
        confidence_note = "The averaged result passes the confidence threshold, but the image-level disagreement should be reviewed."

    st.markdown('<div class="compact-heading">Combined result</div>', unsafe_allow_html=True)
    result_card(
        combined_prediction,
        CONFIDENCE_THRESHOLD,
        heading="Final averaged prediction",
        note=f"{agreement_note} {confidence_note}",
    )

    if combined_confidence < CONFIDENCE_THRESHOLD:
        st.warning("The combined result is still uncertain. Please use a clearer image or consult an expert.")
    elif first_label != second_label:
        st.warning("The two images produced different predictions. Review the averaged result carefully.")
    else:
        st.success("Both images agree.")

    show_top5(combined_prediction["top_k"], "View combined Top-5 predictions", expanded=False)


def main() -> None:
    apply_page_style()
    initialise_session_state()

    header_left, header_right = st.columns([0.82, 0.18])
    with header_left:
        st.markdown('<h1 class="app-title">Caddisfly Image Classifier</h1>', unsafe_allow_html=True)
        st.markdown(
            """
            <div class="app-subtitle">
            Upload a clear image of a caddisfly specimen to receive a predicted class and confidence score.
            </div>
            """,
            unsafe_allow_html=True,
        )
    with header_right:
        st.button("Reset", on_click=reset_application, use_container_width=True)

    try:
        artifacts = get_model_artifacts()
    except Exception as exc:
        show_error("The model could not be loaded. Check that the packaged model files are present.", exc)
        st.stop()

    show_model_information(artifacts)

    try:
        show_first_image_workflow(artifacts)
        show_second_image_workflow(artifacts)
    except InferenceError as exc:
        show_error("Prediction could not be completed for the uploaded image.", exc)
    except Exception as exc:
        show_error("An unexpected application error occurred.", exc)

    st.markdown(
        """
        <div class="footer-note">
        This system provides model-based predictions for educational and research purposes.
        Low-confidence or conflicting results should be verified using clearer images or expert taxonomic assessment.
        </div>
        """,
        unsafe_allow_html=True,
    )


if __name__ == "__main__":
    main()
