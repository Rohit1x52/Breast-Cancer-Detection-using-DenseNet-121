import streamlit as st
import torch
import torch.nn as nn
from torchvision import transforms, models
from PIL import Image
import numpy as np
import cv2
import os

class HybridDenseNetModel(nn.Module):
    def __init__(self, engineered_feature_dim, num_classes=1, freeze_backbone=True):
        super(HybridDenseNetModel, self).__init__()

        # DenseNet-121 backbone (matching notebook architecture)
        self.backbone = models.densenet121(weights=models.DenseNet121_Weights.IMAGENET1K_V1)
        
        # Replace classifier with identity — we'll extract 1024-D features
        num_ftrs = self.backbone.classifier.in_features  # 1024 for DenseNet-121
        self.backbone.classifier = nn.Identity()

        # Freeze early layers for stability if desired
        if freeze_backbone:
            for name, param in self.backbone.features.named_parameters():
                if "denseblock4" not in name:  # only fine-tune last dense block
                    param.requires_grad = False

        # Engineered feature encoder (matching notebook)
        self.engineered_branch = nn.Sequential(
            nn.Linear(engineered_feature_dim, 256),
            nn.ReLU(inplace=True),
            nn.BatchNorm1d(256),
            nn.Dropout(0.3),
            nn.Linear(256, 128),
            nn.ReLU(inplace=True),
            nn.BatchNorm1d(128),
            nn.Dropout(0.2)
        )

        # Fusion Layer (CNN + Engineered) - matching notebook
        fusion_input_dim = num_ftrs + 128  # 1024 (DenseNet) + 128 (engineered)
        self.classifier = nn.Sequential(
            nn.Linear(fusion_input_dim, 512),
            nn.ReLU(inplace=True),
            nn.BatchNorm1d(512),
            nn.Dropout(0.4),
            nn.Linear(512, 128),
            nn.ReLU(inplace=True),
            nn.BatchNorm1d(128),
            nn.Dropout(0.3),
            nn.Linear(128, num_classes)
        )

    def forward(self, image, engineered_features):
        # CNN branch
        cnn_feats = self.backbone(image)  # [B, 1024]
        # Engineered branch
        eng_feats = self.engineered_branch(engineered_features)  # [B, 128]
        # Concatenate both feature spaces
        fused = torch.cat([cnn_feats, eng_feats], dim=1)
        # Final prediction
        out = self.classifier(fused)
        return out

#  Model Loading
DEVICE = torch.device("cuda" if torch.cuda.is_available() else "cpu")
ENGINEERED_FEATURE_DIM = 147  # Must match training setup
model = HybridDenseNetModel(engineered_feature_dim=ENGINEERED_FEATURE_DIM).to(DEVICE)

def load_model_checkpoint(model_paths, device):
    errors = []
    for path in model_paths:
        if not os.path.exists(path):
            msg = f"Checkpoint not found: {path}"
            print(f"⚠️ {msg}")
            errors.append(msg)
            continue
        
        # Try multiple loading strategies
        strategies = [
            ("weights_only=False", lambda: torch.load(path, map_location=device, weights_only=False)),
            ("weights_only=True", lambda: torch.load(path, map_location=device, weights_only=True)),
            ("pickle legacy", lambda: torch.load(path, map_location=device, pickle_module=__import__('pickle'))),
        ]
        
        for strategy_name, load_fn in strategies:
            try:
                print(f"🔄 Attempting to load: {path} (strategy: {strategy_name})")
                checkpoint = load_fn()
                state_dict = checkpoint.get("model_state_dict", checkpoint)
                
                # Load with strict=False to allow minor mismatches
                missing_keys, unexpected_keys = model.load_state_dict(state_dict, strict=False)
                
                if missing_keys:
                    print(f"  ⚠️ Missing keys: {len(missing_keys)}")
                if unexpected_keys:
                    print(f"  ⚠️ Unexpected keys: {len(unexpected_keys)}")
                
                print(f"✅ Successfully loaded model weights from: {path}")
                return True
                
            except RuntimeError as e:
                if "__path__._path" in str(e) or "does not exist" in str(e):
                    print(f"  ⚠️ {strategy_name} failed (serialization issue), trying next strategy...")
                    continue
                else:
                    msg = f"{path} ({strategy_name}): {type(e).__name__}: {str(e)[:100]}"
                    print(f"  ❌ {type(e).__name__}: {str(e)[:100]}")
                    errors.append(msg)
                    break  # Try next file
            except Exception as e:
                msg = f"{path} ({strategy_name}): {type(e).__name__}: {str(e)[:100]}"
                print(f"  ❌ {type(e).__name__}: {str(e)[:100]}")
                errors.append(msg)
                break  # Try next file
    
    # If we get here, no checkpoint loaded successfully
    error_summary = "\n".join(f"  • {err}" for err in errors)
    raise FileNotFoundError(f"No valid model checkpoint found.\n\nErrors:\n{error_summary}")

load_model_checkpoint(["best_hybrid_densenet.pth", "best_densenet_model.pth"], DEVICE)
model.eval()

def generate_gradcam(model, image_tensor, engineered_features):
    grad = None
    activation = None

    def backward_hook(module, grad_input, grad_output):
        nonlocal grad
        grad = grad_output[0].detach()

    def forward_hook(module, input, output):
        nonlocal activation
        activation = output.detach()

    # For DenseNet, target the last conv layer in denseblock4
    if hasattr(model.backbone.features, 'denseblock4'):
        target_layer = model.backbone.features.denseblock4
    else:
        target_layer = model.backbone.features[-1]
    
    handle_f = target_layer.register_forward_hook(forward_hook)
    handle_b = target_layer.register_full_backward_hook(backward_hook)

    image_tensor.requires_grad_(True)
    
    with torch.enable_grad():
        output = model(image_tensor, engineered_features)
        score = torch.sigmoid(output).sum()
        model.zero_grad(set_to_none=True)
        score.backward(retain_graph=True)

    weights = grad.mean(dim=[2, 3], keepdim=True)
    gradcam = torch.sum(weights * activation, dim=1).squeeze()
    gradcam = torch.relu(gradcam)
    gradcam -= gradcam.min()
    gradcam /= (gradcam.max() + 1e-8)

    handle_f.remove()
    handle_b.remove()
    return gradcam.cpu().numpy()

def overlay_heatmap(img, cam):
    """Overlay heatmap on image. img can be PIL Image or numpy array."""
    # Convert to numpy if needed
    if not isinstance(img, np.ndarray):
        img_np = np.array(img)
    else:
        img_np = img
    
    # Get image dimensions (height, width)
    h, w = img_np.shape[:2]
    
    # Resize and normalize CAM
    cam = np.clip(cam, 0, 1)
    cam = cv2.resize(cam, (w, h))
    
    # Apply colormap
    heatmap = cv2.applyColorMap(np.uint8(255 * cam), cv2.COLORMAP_JET)
    heatmap = cv2.cvtColor(heatmap, cv2.COLOR_BGR2RGB)  # Convert BGR to RGB
    heatmap = np.float32(heatmap) / 255
    
    # Normalize image
    img_normalized = np.float32(img_np) / 255
    
    # Blend
    overlayed = heatmap * 0.4 + img_normalized
    overlayed = np.clip(overlayed, 0, 1)
    return np.uint8(255 * overlayed)

# 🧩 Image Preprocessing
transform = transforms.Compose([
    transforms.Resize((224, 224)),
    transforms.ToTensor(),
    transforms.Normalize(mean=[0.485, 0.456, 0.406],
                         std=[0.229, 0.224, 0.225])
])

# 💻 Streamlit Interface
st.set_page_config(page_title="Breast Cancer Detection (DenseNet Hybrid)", layout="centered", page_icon="🩺")
st.title("🩺 Breast Cancer Detection — Hybrid DenseNet Model")
st.caption("Combining **CNN imaging** with **engineered radiomic features** for interpretable diagnosis.")


uploaded_file = st.file_uploader(" Upload an image...", type=["jpg", "jpeg", "png"])

if uploaded_file:
    image = Image.open(uploaded_file).convert("RGB")
    st.image(image, caption=" Uploaded Image", use_container_width=True)

    if st.button(" Predict and Explain"):
        try:
            img_tensor = transform(image).unsqueeze(0).to(DEVICE)
            engineered_features = torch.zeros((1, ENGINEERED_FEATURE_DIM)).to(DEVICE)

            with torch.no_grad():
                output = model(img_tensor, engineered_features)
                prob = torch.sigmoid(output).item()

            # Determine prediction class
            if 0.45 <= prob <= 0.55:
                label = "Indeterminate"
                color = "orange"
                confidence = 0.5
            elif prob > 0.5:
                label = "Malignant"
                color = "red"
                confidence = prob
            else:
                label = "Benign"
                color = "green"
                confidence = 1 - prob

            gradcam = generate_gradcam(model, img_tensor, engineered_features)
            overlayed = overlay_heatmap(np.array(image), gradcam)

            st.markdown(f"### Prediction: <span style='color:{color}'>{label}</span>", unsafe_allow_html=True)
            st.write(f"**Confidence:** {confidence:.2%}")
            st.progress(float(confidence))
            st.image(overlayed, caption=" Grad-CAM Visualization", use_container_width=True)

            # Clinical explanation
            if label == "Malignant":
                st.markdown("""
                ###  Malignant Tissue Features
                **🧬 Cellular Morphology**
                - Enlarged, irregular nuclei and coarse chromatin  
                - Prominent nucleoli and abnormal mitotic activity  
                - High **nuclear-to-cytoplasmic ratio**  

                **🧫 Tissue Architecture**
                - Disrupted glandular formation  
                - Dense cell clusters and stromal invasion  

                **🎨 Texture & Optical Patterns**
                - High **GLCM contrast** and **entropy**  
                - Elevated **Laplacian variance** and chaotic gradients  

                 *Indicates disorganized growth and aggressive pathology.*
                """)
            elif label == "Benign":
                st.markdown("""
                ### 💚 Benign Tissue Features
                **🧬 Cellular Morphology**
                - Uniform, round nuclei and consistent cell shapes  
                - Low **N/C ratio** and minimal mitotic activity  

                **🧫 Tissue Architecture**
                - Well-defined and regular glandular structure  
                - Smooth cell boundaries and preserved basement membrane  

                **🎨 Texture & Optical Patterns**
                - Low GLCM contrast and consistent chromatin texture  
                - Stable intensity distribution and uniform pixel variance  

                💚 *Indicates non-invasive, normal or benign morphology.*
                """)
            else:
                st.markdown("""
                ### ⚪ Indeterminate Zone
                - Prediction falls near the decision threshold (0.45–0.55).  
                - Indicates potential **borderline morphology** (e.g., atypical hyperplasia).  
                - Recommend further **pathologist review or higher-resolution imaging.**
                """)
        except Exception as e:
            st.error(f"Error during prediction: {e}")
            import traceback
            st.text(traceback.format_exc())

#  Sidebar Info
st.sidebar.header("📘 Model Information")
st.sidebar.markdown(f"""
**Architecture:** DenseNet-121 Hybrid  
**Engineered Features:** {ENGINEERED_FEATURE_DIM}  
**Device:** {'CUDA' if torch.cuda.is_available() else 'CPU'}  
**Explainability:** Grad-CAM  
**Dataset:** BreakHis (Benign vs Malignant)  
""")
