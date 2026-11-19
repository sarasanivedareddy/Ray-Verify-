import os
from pathlib import Path
import time

import gradio as gr
import torch
import torch.nn as nn
from PIL import Image, UnidentifiedImageError
from torchvision import transforms
from torchvision.models import DenseNet121_Weights, densenet121
from prometheus_client import Counter, Histogram, Gauge, start_http_server
# Prometheus metrics
PREDICTIONS = Counter(
    "ray_verify_predictions_total",
    "Total number of successful predictions",
    ["prediction"]
)

PREDICTION_DURATION = Histogram(
    "ray_verify_prediction_duration_seconds",
    "Time taken to process a prediction"
)

PREDICTION_CONFIDENCE = Gauge(
    "ray_verify_prediction_confidence",
    "Maximum average ensemble score for the latest prediction"
)

MODEL_DIR = Path(os.environ.get("MODEL_DIR", "/app/saved_models"))
ENSEMBLE_PATH = MODEL_DIR / "densenet121_ensemble.pkl"

DEVICE = torch.device("cuda" if torch.cuda.is_available() else "cpu")
WITHHOLDING_THRESHOLD = 0.80
EXPECTED_MODELS = 5

print(f"Ray Verify device: {DEVICE}")
print(f"Loading ensemble from: {ENSEMBLE_PATH}")

if not ENSEMBLE_PATH.is_file():
    raise FileNotFoundError(
        f"DenseNet ensemble not found at {ENSEMBLE_PATH}. "
        "Place densenet121_ensemble.pkl in the saved_models directory "
        "or set the MODEL_DIR environment variable.")

# Preprocessing

DENSENET_WEIGHTS = DenseNet121_Weights.DEFAULT

EVAL_TRANSFORM = transforms.Compose([
    transforms.Grayscale(num_output_channels=3),
    DENSENET_WEIGHTS.transforms(),])


# Load the five trained DenseNet-121 fold models once at startup

def load_densenet_ensemble(bundle_path: Path):
    checkpoint = torch.load(
        bundle_path,
        map_location="cpu",
        weights_only=True )

    if not isinstance(checkpoint, dict) or "fold_models" not in checkpoint:
        raise ValueError(
            "Unexpected ensemble checkpoint format. "
            "Expected a dictionary containing 'fold_models'."  )

    fold_states = checkpoint["fold_models"]

    if len(fold_states) != EXPECTED_MODELS:
        raise ValueError(
            f"Expected {EXPECTED_MODELS} fold models, "
            f"but found {len(fold_states)}." )

    models = []

    for fold_number, state_dict in enumerate(fold_states, start=1):
        model = densenet121(weights=None)
        model.classifier = nn.Linear(model.classifier.in_features, 2)
        model.load_state_dict(state_dict)
        model.to(DEVICE)
        model.eval()
        models.append(model)

        print(f"Loaded DenseNet fold {fold_number}")

    return models


DENSENET_MODELS = load_densenet_ensemble(ENSEMBLE_PATH)
print(f"DenseNet ensemble ready: {len(DENSENET_MODELS)} models")


# Ensemble inference

def analyze_densenet(image):
    if image is None:
        return (
            "Please upload an image first.",
            "",
            None,
            ""
        )
    start_time = time.perf_counter()
    try:
        if not isinstance(image, Image.Image):
            image = Image.fromarray(image)

        image = image.convert("L")
        input_tensor = EVAL_TRANSFORM(image).unsqueeze(0).to(DEVICE)

        model_probabilities = []

        with torch.inference_mode():
            for model in DENSENET_MODELS:
                logits = model(input_tensor)
                probabilities = torch.softmax(logits, dim=1)
                model_probabilities.append(probabilities)

            ensemble_probabilities = torch.stack(
                model_probabilities
            ).mean(dim=0)[0]

        patient_score, synthetic_score = (
            ensemble_probabilities.detach().cpu().tolist() )
        maximum_score = max(patient_score, synthetic_score)

        if maximum_score < WITHHOLDING_THRESHOLD:
            prediction = "Uncertain"
            guidance = (
                "Classification withheld because the highest average "
                "model score is below 0.80. Check the image's source "
                "and provenance before deciding." )
        else:
            prediction = (
                "Patient-acquired"
                if patient_score >= synthetic_score
                else "Synthetic" )
            guidance = (
                "Review the image's source and provenance. "
                "The model can still be incorrect." )

        result_markdown = f"## {prediction}\n\n{guidance}"

        scores_text = (
            f"Patient-acquired: {patient_score:.4f}\n"
            f"Synthetic: {synthetic_score:.4f}\n"
            f"Withholding threshold: {WITHHOLDING_THRESHOLD:.2f}\n\n"
            "These are uncalibrated model scores, not probabilities "
            "that the prediction is correct."  )

        record = {"prediction": prediction}

        # Update Prometheus metrics
        PREDICTIONS.labels(prediction=prediction).inc()
        PREDICTION_DURATION.observe(time.perf_counter() - start_time)
        PREDICTION_CONFIDENCE.set(maximum_score)

        return result_markdown, scores_text, record, ""

    except (OSError, UnidentifiedImageError, ValueError) as error:
        print(f"Image processing error: {error!r}")
        return (
            "The image could not be processed. Please upload a readable image.",
            "",
            None,
            ""  )
    except Exception as error:
        print(f"DenseNet inference error: {error!r}")
        return (
            "Analysis failed. Please try again or contact the application administrator.",
            "",
            None,
            "")


def record_densenet_decision(record, decision, notes):
    if record is None:
        return "Analyze an image before recording your decision."

    return (
        f"AI prediction: {record['prediction']}\n"
        f"Curator decision: {decision}\n"
        f"Notes: {notes.strip() or 'None'}\n\n"
        "This record is displayed for this session only; "
        "it is not saved to a database."   )


def reset_densenet():
    return (
        None,                       # uploaded image
        "## Ready to analyze",      # prediction
        "",                         # scores
        None,                       # prediction state
        "Leave unresolved",         # curator decision
        "",                         # notes
        ""                          # decision record
   )


# Gradio interface

CSS = """
.gradio-container {
    max-width: 1100px !important;
    margin: auto;}
#dn-header {
    background: linear-gradient(120deg, #17324d, #087f8c);
    padding: 26px;
    border-radius: 16px;
    margin-bottom: 18px;}
#dn-header h1, #dn-header p {
    color: white !important;}
"""

with gr.Blocks(
    title="Ray Verify | DenseNet",
    theme=gr.themes.Soft(),
    css=CSS
) as demo:

    gr.HTML("""
        <div id="dn-header">
            <h1>Ray Verify</h1>
            <p>Chest X-ray origin screening · DenseNet-121 ensemble</p>
        </div>
    """)

    gr.Markdown(
        "Upload a de-identified chest X-ray, review the prediction, "
        "and make your own decision. "
        "**Research prototype—not for medical diagnosis.**" )

    prediction_state = gr.State(None)

    with gr.Row():
        with gr.Column():
            image_input = gr.Image(
                label="Upload chest X-ray",
                type="pil",
                sources=["upload"],
                height=350 )

            with gr.Row():
                analyze_button = gr.Button(
                    "Analyze image",
                    variant="primary"   )
                clear_button = gr.Button("Clear")

        with gr.Column():
            result_display = gr.Markdown("## Ready to analyze")

            scores_display = gr.Textbox(
                label="Average scores from five models",
                lines=6,
                interactive=False)

            gr.Markdown(
                "**Uncertain** means the highest average score is below "
                "0.80. A higher score does not guarantee a correct prediction." )

    gr.Markdown("### Your review")
    gr.Markdown(
        "Your decision remains separate from the AI prediction. "
        "You can disagree with the model or leave the case unresolved." )

    curator_decision = gr.Radio(
        choices=[
            "Include in research dataset",
            "Exclude from research dataset",
            "Leave unresolved"  ],
        value="Leave unresolved",
        label="Curator decision")

    curator_notes = gr.Textbox(
        label="Review notes",
        placeholder="Explain your decision or what needs checking.",
        lines=3)

    record_button = gr.Button("Record session decision")

    decision_summary = gr.Textbox(
        label="Decision record",
        lines=5,
        interactive=False )

    gr.Markdown(
        "Decisions are displayed for this session only and are not saved "
        "to a database. Upload only de-identified research images. "
        "This prototype does not verify whether an uploaded image is a chest X-ray." )

    analyze_button.click(
        fn=analyze_densenet,
        inputs=image_input,
        outputs=[
            result_display,
            scores_display,
            prediction_state,
            decision_summary   ],
        concurrency_limit=1 )

    record_button.click(
        fn=record_densenet_decision,
        inputs=[
            prediction_state,
            curator_decision,
            curator_notes ],
        outputs=decision_summary )

    clear_button.click(
        fn=reset_densenet,
        inputs=[],
        outputs=[
            image_input,
            result_display,
            scores_display,
            prediction_state,
            curator_decision,
            curator_notes,
            decision_summary  ] )


if __name__ == "__main__":

    start_http_server(
        int(os.environ.get("METRICS_PORT", "8000"))
    )

    print("Prometheus metrics available on port 8000", flush=True)

    demo.queue().launch(
        server_name="0.0.0.0",
        server_port=int(os.environ.get("PORT", "7860")),
        share=False,
        max_file_size="10mb"
    )
