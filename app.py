import streamlit as st
import torch
import torch.nn as nn
import librosa
import numpy as np
import json
import tempfile
import plotly.graph_objects as go
import matplotlib.pyplot as plt


class ConvBlock(nn.Module):
    def __init__(self, in_ch, out_ch, stride=2):
        super().__init__()
        self.block = nn.Sequential(
            nn.Conv2d(in_ch, out_ch, kernel_size=3, stride=stride, padding=1, bias=False),
            nn.BatchNorm2d(out_ch),
            nn.LeakyReLU(0.2, inplace=True),
        )
    def forward(self, x):
        return self.block(x)


class DeconvBlock(nn.Module):
    def __init__(self, in_ch, out_ch, stride=2, last=False):
        super().__init__()
        act  = nn.Sigmoid() if last else nn.LeakyReLU(0.2, inplace=True)
        norm = [] if last else [nn.BatchNorm2d(out_ch)]
        self.block = nn.Sequential(
            nn.ConvTranspose2d(in_ch, out_ch, kernel_size=3, stride=stride,
                               padding=1, output_padding=1, bias=False),
            *norm, act,
        )
    def forward(self, x):
        return self.block(x)


class ConvAutoencoder(nn.Module):
    def __init__(self, latent_dim=32):
        super().__init__()
        self.encoder_conv = nn.Sequential(
            ConvBlock(1, 32), ConvBlock(32, 64),
            ConvBlock(64, 128), ConvBlock(128, 128),
        )
        self.encoder_fc   = nn.Linear(128 * 8 * 8, latent_dim)
        self.decoder_fc   = nn.Linear(latent_dim, 128 * 8 * 8)
        self.decoder_conv = nn.Sequential(
            DeconvBlock(128, 128), DeconvBlock(128, 64),
            DeconvBlock(64, 32), DeconvBlock(32, 1, last=True),
        )
    def forward(self, x):
        h = self.encoder_conv(x)
        h = h.view(h.size(0), -1)
        z = self.encoder_fc(h)
        h = self.decoder_fc(z)
        h = h.view(h.size(0), 128, 8, 8)
        return self.decoder_conv(h)


@st.cache_resource
def load_model_and_artifacts():
    with open("artifacts.json") as f:
        arts = json.load(f)
    cfg   = arts["cfg"]
    model = ConvAutoencoder(latent_dim=cfg["latent_dim"])
    model.load_state_dict(torch.load(cfg["model_path"], map_location="cpu"))
    model.eval()
    return model, arts["threshold"], cfg, arts


def wav_to_log_mel(path, cfg):
    y, sr = librosa.load(path, sr=cfg["sample_rate"])
    seg_len  = int(cfg["sample_rate"] * cfg["segment_duration"])
    segments = [y[i:i+seg_len] for i in range(0, len(y) - seg_len + 1, seg_len)]
    specs = []
    for seg in segments:
        if len(seg) < seg_len:
            continue
        S    = librosa.feature.melspectrogram(y=seg, sr=sr,
                   n_mels=cfg["n_mels"], n_fft=cfg["n_fft"], hop_length=cfg["hop_length"])
        S_db = librosa.power_to_db(S, ref=np.max)
        if S_db.shape[1] != cfg["spec_width"]:
            S_db = np.array([
                np.interp(np.linspace(0, S_db.shape[1]-1, cfg["spec_width"]),
                          np.arange(S_db.shape[1]), S_db[i])
                for i in range(S_db.shape[0])
            ])
        S_norm = (S_db - S_db.min()) / (S_db.max() - S_db.min() + 1e-8)
        specs.append(S_norm[np.newaxis])
    return np.stack(specs).astype(np.float32) if specs else None


def predict(model, specs_np):
    t = torch.from_numpy(specs_np)
    with torch.no_grad():
        recon = model(t)
    mse_per_seg = ((recon - t) ** 2).mean(dim=[1, 2, 3]).numpy()
    return recon.numpy(), mse_per_seg


def gauge_chart(score, threshold):
    color = "#e74c3c" if score > threshold else "#2ecc71"
    fig   = go.Figure(go.Indicator(
        mode  = "gauge+number+delta",
        value = score,
        delta = {"reference": threshold, "valueformat": ".5f"},
        title = {"text": "Anomaly Score (MSE)"},
        number= {"valueformat": ".5f"},
        gauge = {
            "axis" : {"range": [0, threshold * 2]},
            "bar"  : {"color": color},
            "steps": [
                {"range": [0, threshold],          "color": "#d5f5e3"},
                {"range": [threshold, threshold*2], "color": "#fadbd8"},
            ],
            "threshold": {
                "line" : {"color": "black", "width": 3},
                "thickness": 0.8,
                "value": threshold,
            },
        },
    ))
    fig.update_layout(height=300, margin=dict(t=40, b=10, l=20, r=20))
    return fig


st.set_page_config(page_title="Industrial Sound Anomaly Detector", layout="wide")
st.title("Industrial Sound Anomaly Detector")
st.caption("Convolutional Autoencoder on MIMII Pump dataset - unsupervised anomaly detection")

model, threshold, cfg, arts = load_model_and_artifacts()

with st.sidebar:
    st.header("Model Info")
    st.metric("ROC-AUC", f"{arts['auc']:.4f}")
    st.metric("Threshold", f"{threshold:.6f}")
    st.metric("Latent Dim", cfg["latent_dim"])
    st.metric("Epochs Trained", cfg["epochs"])
    st.metric("Loss Function", "MSE + SSIM")
    st.metric("Machine ID", cfg["machine_id"])

uploaded = st.file_uploader("Upload a WAV file from a pump machine", type=["wav"])

if uploaded is not None:
    with tempfile.NamedTemporaryFile(suffix=".wav", delete=False) as tmp:
        tmp.write(uploaded.read())
        tmp_path = tmp.name

    st.audio(uploaded)

    specs = wav_to_log_mel(tmp_path, cfg)
    if specs is None:
        st.error("Could not extract spectrogram - file too short or invalid.")
        st.stop()

    recon_np, mse_segs = predict(model, specs)
    avg_score          = float(mse_segs.mean())
    is_anomaly         = avg_score > threshold

    if is_anomaly:
        st.error(f"ANOMALY DETECTED - Score: {avg_score:.6f} (threshold: {threshold:.6f})")
    else:
        st.success(f"NORMAL OPERATION - Score: {avg_score:.6f} (threshold: {threshold:.6f})")

    col1, col2 = st.columns([2, 1])

    with col1:
        orig     = specs[0, 0]
        rec      = recon_np[0, 0]
        residual = np.abs(orig - rec)
        fig, axes = plt.subplots(1, 3, figsize=(14, 3.5))
        for ax, data, title, cmap in zip(
            axes,
            [orig, rec, residual],
            ["Input Spectrogram", "Reconstructed", "Residual"],
            ["viridis", "viridis", "magma"],
        ):
            im = ax.imshow(data, aspect="auto", origin="lower", cmap=cmap, vmin=0, vmax=1)
            ax.set_title(title)
            ax.set_xlabel("Time Frame")
            ax.set_ylabel("Mel Bin")
            plt.colorbar(im, ax=ax)
        plt.suptitle(f"Segment 1 | MSE = {mse_segs[0]:.6f}", y=1.02)
        plt.tight_layout()
        st.pyplot(fig, use_container_width=True)

    with col2:
        st.plotly_chart(gauge_chart(avg_score, threshold), use_container_width=True)

    if len(mse_segs) > 1:
        st.subheader("Per-Segment Anomaly Scores")
        fig_seg = go.Figure()
        fig_seg.add_trace(go.Bar(
            x=list(range(1, len(mse_segs)+1)), y=mse_segs.tolist(),
            marker_color=["#e74c3c" if s > threshold else "#2ecc71" for s in mse_segs],
        ))
        fig_seg.add_hline(y=threshold, line_dash="dash", line_color="black",
                          annotation_text="Threshold")
        fig_seg.update_layout(xaxis_title="Segment", yaxis_title="MSE",
                               height=300, margin=dict(t=20, b=20))
        st.plotly_chart(fig_seg, use_container_width=True)
