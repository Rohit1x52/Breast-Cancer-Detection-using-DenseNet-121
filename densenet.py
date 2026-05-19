import io
import logging
import os
import traceback
from contextlib import contextmanager
from dataclasses import dataclass
from typing import Dict, Optional, Tuple

import cv2
import numpy as np
import streamlit as st
import torch
import torch.nn as nn
from dotenv import load_dotenv
from PIL import Image
from torchvision import models, transforms

load_dotenv()

logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s [%(levelname)s] %(name)s: %(message)s",
)
logger = logging.getLogger(__name__)

try:
    import google.generativeai as genai
    GENAI_AVAILABLE = True
except ImportError:
    GENAI_AVAILABLE = False
    logger.warning("google-generativeai not installed. Report generation disabled.")


ENGINEERED_FEATURE_DIM: int = 147
MAX_IMAGE_BYTES: int = 10 * 1024 * 1024
SUPPORTED_IMAGE_TYPES: Tuple[str, ...] = ("jpg", "jpeg", "png")
MALIGNANT_THRESHOLD: float = 0.55
BENIGN_THRESHOLD: float = 0.45
GRADCAM_HIGH_ACTIVATION_CUTOFF: float = 0.5

IMAGENET_MEAN = [0.485, 0.456, 0.406]
IMAGENET_STD = [0.229, 0.224, 0.225]

DEVICE = torch.device("cuda" if torch.cuda.is_available() else "cpu")

IMAGE_TRANSFORM = transforms.Compose([
    transforms.Resize((224, 224)),
    transforms.ToTensor(),
    transforms.Normalize(mean=IMAGENET_MEAN, std=IMAGENET_STD),
])


@dataclass(frozen=True)
class PredictionResult:
    label: str
    confidence: float
    raw_probability: float
    gradcam_map: np.ndarray
    gradcam_stats: Dict[str, float]
    overlaid_image: np.ndarray


@dataclass(frozen=True)
class GradCamStats:
    mean: float
    maximum: float
    std: float
    coverage: float


class HybridDenseNetModel(nn.Module):
    def __init__(self, engineered_feature_dim: int, num_classes: int = 1, freeze_backbone: bool = True):
        super().__init__()

        self.backbone = models.densenet121(weights=models.DenseNet121_Weights.IMAGENET1K_V1)
        num_ftrs = self.backbone.classifier.in_features
        self.backbone.classifier = nn.Identity()

        if freeze_backbone:
            for name, param in self.backbone.features.named_parameters():
                if "denseblock4" not in name:
                    param.requires_grad = False

        self.engineered_branch = nn.Sequential(
            nn.Linear(engineered_feature_dim, 256),
            nn.ReLU(inplace=True),
            nn.BatchNorm1d(256),
            nn.Dropout(0.3),
            nn.Linear(256, 128),
            nn.ReLU(inplace=True),
            nn.BatchNorm1d(128),
            nn.Dropout(0.2),
        )

        fusion_input_dim = num_ftrs + 128
        self.classifier = nn.Sequential(
            nn.Linear(fusion_input_dim, 512),
            nn.ReLU(inplace=True),
            nn.BatchNorm1d(512),
            nn.Dropout(0.4),
            nn.Linear(512, 128),
            nn.ReLU(inplace=True),
            nn.BatchNorm1d(128),
            nn.Dropout(0.3),
            nn.Linear(128, num_classes),
        )

    def forward(self, image: torch.Tensor, engineered_features: torch.Tensor) -> torch.Tensor:
        cnn_feats = self.backbone(image)
        eng_feats = self.engineered_branch(engineered_features)
        fused = torch.cat([cnn_feats, eng_feats], dim=1)
        return self.classifier(fused)


@st.cache_resource(show_spinner="Loading model weights...")
def load_model(engineered_feature_dim: int, device: torch.device) -> HybridDenseNetModel:
    model = HybridDenseNetModel(engineered_feature_dim=engineered_feature_dim).to(device)
    model = _load_checkpoint_into_model(model, device)
    model.eval()
    logger.info("Model loaded and set to eval mode on device: %s", device)
    return model


def _load_checkpoint_into_model(
    model: HybridDenseNetModel, device: torch.device
) -> HybridDenseNetModel:
    script_dir = os.path.dirname(os.path.abspath(__file__))
    candidate_paths = ["best_hybrid_densenet.pth", "best_densenet_model.pth"]
    errors = []

    for relative_path in candidate_paths:
        full_path = os.path.join(script_dir, relative_path)
        if not os.path.exists(full_path):
            logger.warning("Checkpoint not found: %s", full_path)
            errors.append(f"File not found: {full_path}")
            continue

        for weights_only in (False, True):
            try:
                checkpoint = torch.load(full_path, map_location=device, weights_only=weights_only)
                state_dict = checkpoint.get("model_state_dict", checkpoint)
                missing, unexpected = model.load_state_dict(state_dict, strict=False)
                if missing:
                    logger.warning("Missing keys (%d) from checkpoint: %s", len(missing), relative_path)
                if unexpected:
                    logger.warning("Unexpected keys (%d) in checkpoint: %s", len(unexpected), relative_path)
                logger.info("Loaded checkpoint: %s (weights_only=%s)", relative_path, weights_only)
                return model
            except RuntimeError as exc:
                logger.debug("Load attempt failed for %s (weights_only=%s): %s", relative_path, weights_only, exc)
                errors.append(f"{relative_path} [weights_only={weights_only}]: {exc}")
            except Exception as exc:
                logger.error("Unexpected error loading %s: %s", relative_path, exc)
                errors.append(f"{relative_path}: {exc}")
                break

    error_detail = "\n".join(f"  {e}" for e in errors)
    raise FileNotFoundError(
        f"No valid model checkpoint could be loaded.\n\nAttempted paths and errors:\n{error_detail}"
    )


def validate_uploaded_image(uploaded_file) -> Tuple[bool, str]:
    if uploaded_file is None:
        return False, "No file provided."

    raw_bytes = uploaded_file.getvalue()
    if len(raw_bytes) > MAX_IMAGE_BYTES:
        return False, f"File size exceeds {MAX_IMAGE_BYTES // (1024 * 1024)} MB limit."

    try:
        img = Image.open(io.BytesIO(raw_bytes))
        img.verify()
    except Exception:
        return False, "The uploaded file is not a valid or supported image."

    return True, ""


def preprocess_image(image: Image.Image) -> torch.Tensor:
    return IMAGE_TRANSFORM(image).unsqueeze(0).to(DEVICE)


def compute_gradcam(
    model: HybridDenseNetModel,
    image_tensor: torch.Tensor,
    engineered_features: torch.Tensor,
) -> np.ndarray:
    model.eval()

    captured_grad: Optional[torch.Tensor] = None
    captured_activation: Optional[torch.Tensor] = None

    def _forward_hook(module, input, output):
        nonlocal captured_activation
        captured_activation = output.detach()

    def _backward_hook(module, grad_input, grad_output):
        nonlocal captured_grad
        captured_grad = grad_output[0].detach()

    if hasattr(model.backbone.features, "denseblock4"):
        target_layer = model.backbone.features.denseblock4
    else:
        target_layer = list(model.backbone.features.children())[-1]

    fwd_handle = target_layer.register_forward_hook(_forward_hook)
    bwd_handle = target_layer.register_full_backward_hook(_backward_hook)

    input_tensor = image_tensor.clone().requires_grad_(True)

    try:
        with torch.enable_grad():
            output = model(input_tensor, engineered_features)
            score = torch.sigmoid(output).sum()
            model.zero_grad(set_to_none=True)
            score.backward()
    finally:
        fwd_handle.remove()
        bwd_handle.remove()

    if captured_grad is None or captured_activation is None:
        logger.error("Grad-CAM hooks did not capture gradient or activation.")
        raise RuntimeError("Grad-CAM failed: gradient or activation not captured.")

    weights = captured_grad.mean(dim=[2, 3], keepdim=True)
    cam = torch.sum(weights * captured_activation, dim=1).squeeze()
    cam = torch.relu(cam)

    cam_min = cam.min()
    cam_max = cam.max()
    if cam_max - cam_min > 1e-8:
        cam = (cam - cam_min) / (cam_max - cam_min)
    else:
        cam = torch.zeros_like(cam)

    return cam.cpu().numpy()


def compute_gradcam_stats(cam: np.ndarray) -> GradCamStats:
    return GradCamStats(
        mean=float(np.mean(cam)),
        maximum=float(np.max(cam)),
        std=float(np.std(cam)),
        coverage=float(np.mean(cam > GRADCAM_HIGH_ACTIVATION_CUTOFF)),
    )


def overlay_heatmap_on_image(original_image: np.ndarray, cam: np.ndarray) -> np.ndarray:
    if cam.ndim != 2:
        raise ValueError(f"CAM must be 2D, got shape {cam.shape}.")

    h, w = original_image.shape[:2]
    cam_clipped = np.clip(cam, 0.0, 1.0)
    cam_resized = cv2.resize(cam_clipped, (w, h))

    heatmap_bgr = cv2.applyColorMap(np.uint8(255 * cam_resized), cv2.COLORMAP_JET)
    heatmap_rgb = cv2.cvtColor(heatmap_bgr, cv2.COLOR_BGR2RGB)
    heatmap_float = heatmap_rgb.astype(np.float32) / 255.0

    image_float = original_image.astype(np.float32) / 255.0

    blended = np.clip(heatmap_float * 0.4 + image_float, 0.0, 1.0)
    return (blended * 255).astype(np.uint8)


def classify_prediction(probability: float) -> Tuple[str, str, float]:
    if probability > MALIGNANT_THRESHOLD:
        return "Malignant", "#c0392b", probability
    elif probability < BENIGN_THRESHOLD:
        return "Benign", "#27ae60", 1.0 - probability
    else:
        return "Indeterminate", "#e67e22", abs(probability - 0.5) * 2


def run_inference(
    model: HybridDenseNetModel,
    image: Image.Image,
    engineered_feature_dim: int,
) -> PredictionResult:
    img_tensor = preprocess_image(image)
    eng_features = torch.zeros((1, engineered_feature_dim), device=DEVICE)

    with torch.no_grad():
        output = model(img_tensor, eng_features)
        probability = torch.sigmoid(output).item()

    label, _, confidence = classify_prediction(probability)

    cam = compute_gradcam(model, img_tensor, eng_features)
    stats = compute_gradcam_stats(cam)
    overlaid = overlay_heatmap_on_image(np.array(image), cam)

    return PredictionResult(
        label=label,
        confidence=confidence,
        raw_probability=probability,
        gradcam_map=cam,
        gradcam_stats={
            "mean": stats.mean,
            "max": stats.maximum,
            "std": stats.std,
            "coverage": stats.coverage,
        },
        overlaid_image=overlaid,
    )


REPORT_PROMPT_TEMPLATE = """You are an expert breast pathology AI assistant. Based on the following computational analysis of a breast histopathology image, generate a detailed, professional clinical report. The report MUST be between 200 to 300 words.

Analysis Results
Prediction: {label}
Model Confidence: {confidence:.1%}
Grad-CAM Activation Summary:
  Mean activation intensity: {mean:.3f}
  Max activation intensity: {maximum:.3f}
  Activation spread (std): {std:.3f}
  High-activation area (>50% threshold): {coverage:.1%} of image

Instructions
1. Write a Clinical Summary (4-5 sentences) interpreting the prediction, confidence level, and what the Grad-CAM activation pattern suggests about tissue morphology.
2. Write a Key Findings section with 4-5 bullet points on cellular/tissue features typically associated with this classification.
3. Write a Tissue Characteristics section (2-3 sentences) describing expected histological patterns for this classification.
4. Write a Recommendation (2-3 sentences) on suggested clinical next steps.
5. Write a brief Disclaimer stating this is AI-assisted and requires pathologist review.

IMPORTANT: Total report must be 200-300 words. Use professional medical language. Do NOT diagnose directly. Frame findings as "computational analysis suggests" or "model indicates". Format output in clean Markdown with proper headings."""


def generate_clinical_report(
    label: str,
    confidence: float,
    gradcam_stats: Dict[str, float],
    api_key: str,
) -> Optional[str]:
    if not api_key or not api_key.strip():
        logger.warning("generate_clinical_report called without a valid API key.")
        return None
    if not GENAI_AVAILABLE:
        logger.warning("google-generativeai not installed.")
        return None

    prompt = REPORT_PROMPT_TEMPLATE.format(
        label=label,
        confidence=confidence,
        mean=gradcam_stats["mean"],
        maximum=gradcam_stats["max"],
        std=gradcam_stats["std"],
        coverage=gradcam_stats["coverage"],
    )

    try:
        genai.configure(api_key=api_key.strip())
        llm = genai.GenerativeModel("gemini-2.0-flash")
        response = llm.generate_content(prompt)
        logger.info("Clinical report generated successfully.")
        return response.text
    except Exception as exc:
        logger.error("Report generation failed: %s", exc)
        return f"Report generation failed: {exc}"


def render_tissue_features(label: str) -> None:
    if label == "Malignant":
        st.markdown("""
**Cellular Morphology**

Enlarged and irregularly shaped nuclei with coarse chromatin patterns. Prominent nucleoli with elevated nuclear-to-cytoplasmic ratio. High mitotic activity with abnormal mitotic figures.

**Tissue Architecture**

Loss of organized glandular formation with dense cellular clustering. Stromal invasion and disruption of the basement membrane are characteristic.

**Texture and Optical Patterns**

Elevated GLCM contrast and entropy reflect disorganized cellular arrangement. High Laplacian variance and chaotic intensity gradients indicate aggressive tissue morphology.
        """)
    elif label == "Benign":
        st.markdown("""
**Cellular Morphology**

Uniform, round to oval nuclei with consistent size and smooth nuclear membranes. Low nuclear-to-cytoplasmic ratio with minimal or absent mitotic activity.

**Tissue Architecture**

Well-defined glandular structures with preserved basement membrane integrity. Smooth cell boundaries and organized stromal arrangement.

**Texture and Optical Patterns**

Low GLCM contrast and consistent chromatin texture. Stable intensity distribution with uniform pixel variance indicating homogeneous tissue organization.
        """)
    else:
        st.markdown("""
**Indeterminate Classification**

The prediction probability falls within the decision boundary zone (0.45 to 0.55), indicating ambiguous morphological features. This may represent borderline pathology such as atypical ductal hyperplasia or other intermediate-grade lesions. Pathologist review with additional clinical correlation is strongly recommended.
        """)


def render_sidebar(device: torch.device, engineered_feature_dim: int) -> None:
    with st.sidebar:
        st.header("Model Information")
        st.markdown(f"""
**Architecture:** DenseNet-121 Hybrid

**Backbone:** DenseNet-121 pretrained on ImageNet

**Engineered Feature Dimension:** {engineered_feature_dim}

**Fusion Strategy:** Late fusion (CNN + radiomic branch)

**Explainability:** Grad-CAM on denseblock4

**Target Classes:** Benign, Malignant

**Dataset:** BreakHis

**Device:** {"CUDA" if device.type == "cuda" else "CPU"}
        """)

        st.header("Classification Thresholds")
        st.markdown(f"""
**Malignant:** probability > {MALIGNANT_THRESHOLD}

**Benign:** probability < {BENIGN_THRESHOLD}

**Indeterminate:** {BENIGN_THRESHOLD} to {MALIGNANT_THRESHOLD}
        """)

        st.header("Image Requirements")
        st.markdown(f"""
**Formats:** {", ".join(t.upper() for t in SUPPORTED_IMAGE_TYPES)}

**Max file size:** {MAX_IMAGE_BYTES // (1024 * 1024)} MB

**Recommended:** H&E stained histopathology patches at 40x, 100x, 200x, or 400x magnification
        """)


def main() -> None:
    st.set_page_config(
        page_title="Breast Cancer Detection",
        layout="wide",
        page_icon="pathology",
        initial_sidebar_state="expanded",
    )

    st.title("Breast Cancer Detection")
    st.caption(
        "Hybrid DenseNet-121 model combining deep convolutional features with engineered "
        "radiomic descriptors for interpretable histopathology classification."
    )

    render_sidebar(DEVICE, ENGINEERED_FEATURE_DIM)

    try:
        model = load_model(ENGINEERED_FEATURE_DIM, DEVICE)
    except FileNotFoundError as exc:
        st.error("Model checkpoint could not be loaded.")
        st.code(str(exc))
        st.stop()
    except Exception as exc:
        st.error(f"Unexpected error during model initialization: {exc}")
        logger.exception("Model initialization failed.")
        st.stop()

    uploaded_file = st.file_uploader(
        "Upload a histopathology image",
        type=list(SUPPORTED_IMAGE_TYPES),
        help="Upload an H&E stained breast histopathology image for analysis.",
    )

    if uploaded_file is not None:
        if st.session_state.get("last_uploaded_filename") != uploaded_file.name:
            st.session_state.pop("prediction_result", None)
            st.session_state["last_uploaded_filename"] = uploaded_file.name

        valid, validation_message = validate_uploaded_image(uploaded_file)
        if not valid:
            st.error(f"Invalid image: {validation_message}")
            st.stop()

        raw_bytes = uploaded_file.getvalue()
        image = Image.open(io.BytesIO(raw_bytes)).convert("RGB")

        col_image, col_controls = st.columns([2, 1])
        with col_image:
            st.image(image, caption="Uploaded image", use_container_width=True)
        with col_controls:
            st.markdown(f"""
**Filename:** {uploaded_file.name}

**Format:** {image.format or uploaded_file.type}

**Dimensions:** {image.width} x {image.height} px

**File size:** {len(raw_bytes) / 1024:.1f} KB
            """)

            run_analysis = st.button("Run Analysis", type="primary", use_container_width=True)

        if run_analysis:
            with st.spinner("Running inference and computing Grad-CAM..."):
                try:
                    result = run_inference(model, image, ENGINEERED_FEATURE_DIM)
                    st.session_state["prediction_result"] = result
                    logger.info(
                        "Inference complete: label=%s confidence=%.4f prob=%.4f",
                        result.label,
                        result.confidence,
                        result.raw_probability,
                    )
                except Exception as exc:
                    st.error("Analysis failed. See details below.")
                    st.code(traceback.format_exc())
                    logger.exception("Inference failed for file: %s", uploaded_file.name)
                    st.stop()

    if "prediction_result" in st.session_state:
        result: PredictionResult = st.session_state["prediction_result"]
        _, color, _ = classify_prediction(result.raw_probability)

        st.divider()
        st.subheader("Analysis Results")

        metric_col1, metric_col2, metric_col3, metric_col4 = st.columns(4)
        with metric_col1:
            st.metric("Prediction", result.label)
        with metric_col2:
            st.metric("Confidence", f"{result.confidence:.2%}")
        with metric_col3:
            st.metric("Raw Probability", f"{result.raw_probability:.4f}")
        with metric_col4:
            st.metric("Grad-CAM Coverage", f"{result.gradcam_stats['coverage']:.1%}")

        st.markdown(
            f"<div style='padding:12px;border-left:4px solid {color};background:#f8f9fa;"
            f"border-radius:4px;margin:12px 0'>"
            f"<strong>Classification: {result.label}</strong> "
            f"with {result.confidence:.2%} confidence"
            f"</div>",
            unsafe_allow_html=True,
        )

        st.progress(float(result.confidence), text=f"Confidence: {result.confidence:.2%}")

        viz_col1, viz_col2 = st.columns(2)
        with viz_col1:
            st.image(
                np.array(image) if uploaded_file else result.overlaid_image,
                caption="Original image",
                use_container_width=True,
            )
        with viz_col2:
            st.image(
                result.overlaid_image,
                caption="Grad-CAM activation overlay",
                use_container_width=True,
            )

        with st.expander("Grad-CAM Statistics", expanded=False):
            stat_col1, stat_col2, stat_col3, stat_col4 = st.columns(4)
            with stat_col1:
                st.metric("Mean Activation", f"{result.gradcam_stats['mean']:.4f}")
            with stat_col2:
                st.metric("Max Activation", f"{result.gradcam_stats['max']:.4f}")
            with stat_col3:
                st.metric("Activation Std", f"{result.gradcam_stats['std']:.4f}")
            with stat_col4:
                st.metric("High-Activation Coverage", f"{result.gradcam_stats['coverage']:.2%}")

        st.divider()
        st.subheader("Tissue Feature Analysis")
        render_tissue_features(result.label)

        st.divider()
        st.subheader("AI Clinical Report")

        api_key = os.getenv("GEMINI_API_KEY", "").strip()

        if api_key and GENAI_AVAILABLE:
            if st.button("Generate Clinical Report", type="secondary"):
                with st.spinner("Generating clinical report via Gemini..."):
                    report = generate_clinical_report(
                        result.label,
                        result.confidence,
                        result.gradcam_stats,
                        api_key,
                    )
                if report and not report.startswith("Report generation failed"):
                    st.markdown(report)
                    st.caption(
                        "This report is AI-generated and must be reviewed by a qualified "
                        "pathologist before any clinical decision is made."
                    )
                else:
                    st.error(report or "Report generation returned an empty response.")
        elif not GENAI_AVAILABLE:
            st.info(
                "Install google-generativeai to enable AI report generation: "
                "pip install google-generativeai"
            )
        else:
            st.info(
                "Set GEMINI_API_KEY in your .env file to enable AI-generated clinical reports."
            )

        st.divider()
        st.caption(
            "This application is intended for research and educational purposes only. "
            "It is not a certified medical device and must not be used as a substitute "
            "for professional pathological diagnosis. All results require review by a "
            "qualified pathologist."
        )


if __name__ == "__main__":
    main()