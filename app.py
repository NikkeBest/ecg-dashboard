import streamlit as st
import numpy as np
import pandas as pd
import plotly.graph_objects as go
import tensorflow as tf

@st.cache_resource
def load_model():
    return tf.keras.models.load_model(
        "heartbeat_cnn.keras",
        compile=False
    )

# Load the trained CNN model
model = load_model()

# -----------------------------
# 1. PAGE CONFIGURATION
# -----------------------------
st.set_page_config(
    page_title="ECG Monitor",
    page_icon="💓",
    layout="wide",
    initial_sidebar_state="expanded"
)

# -----------------------------
# 2. MEDICAL DASHBOARD STYLING
# -----------------------------
st.markdown("""
<style>
    .stApp {
        background-color: #F4F7FB;
    }
    [data-testid="stSidebar"] {
        background-color: #000000;
        border-right: 1px solid #E2E8F0;
    }
    .block-container {
        padding-top: 1.8rem;
        padding-bottom: 2rem;
    }
    .main-title {
        color: #17365D;
        font-size: 30px;
        font-weight: 750;
        margin-bottom: 3px;
    }
    .subtitle {
        color: #64748B;
        font-size: 14px;
        margin-bottom: 22px;
    }
    .section-title {
        color: #17365D;
        font-size: 19px;
        font-weight: 700;
        margin-top: 10px;
    }
    .notice {
        background: #FFF7E6;
        color: #8A5A00;
        border: 1px solid #F5D58A;
        padding: 12px 16px;
        border-radius: 10px;
        font-size: 13px;
    }
    div[data-testid="stMetric"] {
        background: white;
        border: 1px solid #E2E8F0;
        padding: 16px;
        border-radius: 12px;
        box-shadow: 0 2px 7px rgba(30, 50, 80, 0.04);
    }
    div[data-testid="stMetricLabel"] {
        color: #64748B;
    }
    div[data-testid="stMetricValue"] {
        color: #17365D;
    }
</style>
""", unsafe_allow_html=True)

# -----------------------------
# 3. SAMPLE DATA
# -----------------------------
# These values are simulated for UI development.
# They are NOT real patient readings or model predictions.

rng = np.random.default_rng(42)

time = np.linspace(0, 5, 1500)
ecg = 0.025 * np.sin(2 * np.pi * 0.3 * time)

# Add illustrative heartbeat-like waveforms
for beat in np.arange(0.5, 5.0, 0.8):
    ecg += 0.12 * np.exp(-((time - (beat - 0.18)) / 0.045) ** 2)
    ecg -= 0.16 * np.exp(-((time - (beat - 0.035)) / 0.012) ** 2)
    ecg += 1.00 * np.exp(-((time - beat) / 0.015) ** 2)
    ecg -= 0.25 * np.exp(-((time - (beat + 0.035)) / 0.018) ** 2)
    ecg += 0.30 * np.exp(-((time - (beat + 0.22)) / 0.075) ** 2)

ecg += rng.normal(0, 0.012, len(time))

# Illustrative model results
category = "N — Normal"
confidence = 96.0
heart_rate = 75

# -----------------------------
# 4. SIDEBAR
# -----------------------------
with st.sidebar:
    st.markdown("## 💓 ECG Monitor")
    st.caption("Heartbeat Classification System")
    st.divider()

    page = st.radio(
        "NAVIGATION",
        [
            "Overview",
            "ECG Waveform",
            "Classification Demo",
            "Model Evaluation",
            "Classification History"
        ]
    )

    st.divider()
    st.markdown("### System status")
    st.success("Dashboard running")
    st.info("Demo data mode")

    st.caption("Hardware and AI model integration pending.")

# -----------------------------
# 5. DASHBOARD HEADER
# -----------------------------
st.markdown(
    '<div class="main-title">ECG Heartbeat Dashboard</div>',
    unsafe_allow_html=True
)
st.markdown(
    '<div class="subtitle">'
    'ECG signal visualization, heartbeat classification and model evaluation'
    '</div>',
    unsafe_allow_html=True
)

st.markdown(
    '<div class="notice">'
    'DEMO MODE — All readings, predictions and evaluation scores on this '
    'page are simulated. They must not be used for medical decisions.'
    '</div>',
    unsafe_allow_html=True
)

st.write("")

# -----------------------------
# 6. REUSABLE ECG CHART
# -----------------------------
def make_ecg_chart():
    fig = go.Figure()

    fig.add_trace(
        go.Scatter(
            x=time,
            y=ecg,
            mode="lines",
            name="ECG signal",
            line=dict(color="#0D9488", width=1.7),
            hovertemplate="Time: %{x:.2f} s<br>"
                          "Amplitude: %{y:.3f}<extra></extra>"
        )
    )

    fig.update_layout(
        height=330,
        margin=dict(l=15, r=15, t=15, b=15),
        paper_bgcolor="white",
        plot_bgcolor="white",
        xaxis_title="Time (seconds)",
        yaxis_title="Illustrative amplitude",
        font=dict(color="#475569"),
        showlegend=False
    )

    fig.update_xaxes(
        showgrid=True,
        gridcolor="#FFFFFF",
        zeroline=False
    )
    fig.update_yaxes(
        showgrid=True,
        gridcolor="#FFFFFF",
        zeroline=False
    )

    return fig

# -----------------------------
# 7. OVERVIEW PAGE
# -----------------------------
if page == "Overview":

    c1, c2, c3 = st.columns(3)

    c1.metric("Predicted Category", category)
    c2.metric("Demo Confidence", f"{confidence:.1f}%")
    c3.metric("Illustrative Heart Rate", f"{heart_rate} BPM")

    st.markdown(
        '<div class="section-title">ECG Signal Preview</div>',
        unsafe_allow_html=True
    )
    st.plotly_chart(
        make_ecg_chart(),
        use_container_width=True,
        key="overview_ecg"
    )

    left, right = st.columns(2)

    with left:
        st.markdown("### Classification Result")
        st.markdown("**Predicted class:** N — Normal")
        st.progress(confidence / 100)
        st.caption("Illustrative model score, not a validated probability.")

    with right:
        st.markdown("### General Next Step")
        st.info(
            "Demo message: follow appropriate healthcare guidance. "
            "A real system must not make a diagnosis from this label alone."
        )

    st.markdown("### Recent Demo Records")

    history = pd.DataFrame({
        "Record ID": ["DEMO-001", "DEMO-002", "DEMO-003"],
        "Predicted Class": ["N", "V", "N"],
        "Score": ["96%", "88%", "93%"],
        "Data Source": ["Simulated"] * 3
    })

    st.dataframe(
        history,
        hide_index=True,
        use_container_width=True
    )


# -----------------------------
# 8. ECG WAVEFORM PAGE
# -----------------------------
elif page == "ECG Waveform":

    st.markdown("### ECG Signal Analysis")
    st.write(
        "Upload a CSV containing ECG samples to visualize "
        "the signal. You can also use the simulated demo signal."
    )

    uploaded_file = st.file_uploader(
        "Upload ECG CSV file",
        type=["csv"],
        help="Use a CSV with a numerical ECG signal column."
    )

    if uploaded_file is not None:
        try:
            df = pd.read_csv(uploaded_file)

            if df.empty or len(df.columns) == 0:
                st.error("The uploaded CSV is empty.")
                st.stop()

            st.write("### Preview of uploaded data")
            st.dataframe(df.head(10), use_container_width=True)

            # Find columns containing numeric ECG samples
            numeric_columns = [
                col for col in df.columns
                if pd.to_numeric(df[col], errors="coerce").notna().any()
            ]

            if not numeric_columns:
                st.error(
                    "No numerical columns were found. "
                    "Please upload a CSV containing ECG samples."
                )
                st.stop()

            # Let the user select the signal column
            signal_column = st.selectbox(
                "Select the ECG signal column",
                numeric_columns
            )

            signal = pd.to_numeric(
                df[signal_column], errors="coerce"
            ).dropna().to_numpy()

            if len(signal) < 2:
                st.error("The selected column has too few valid samples.")
                st.stop()

            sample_rate = st.number_input(
                "Sampling frequency (Hz)",
                min_value=1,
                max_value=5000,
                value=360,
                step=1,
                help=(
                    "Use the actual sampling rate specified by "
                    "your dataset or acquisition system."
                )
            )

            # Convert sample positions to time
            signal_time = np.arange(len(signal)) / sample_rate

            fig = go.Figure()
            fig.add_trace(
                go.Scatter(
                    x=signal_time,
                    y=signal,
                    mode="lines",
                    name="Uploaded ECG",
                    line=dict(color="#0D9488", width=1.5)
                )
            )

            fig.update_layout(
                title="Uploaded ECG Waveform",
                xaxis_title="Time (seconds)",
                yaxis_title="Signal amplitude (original units)",
                height=400
            )

            st.plotly_chart(fig, use_container_width=True)

            col1, col2, col3 = st.columns(3)
            col1.metric("Valid Samples", f"{len(signal):,}")
            col2.metric("Sampling Rate", f"{sample_rate} Hz")
            col3.metric(
                "Signal Duration",
                f"{len(signal) / sample_rate:.2f} s"
            )

            st.success(
                "ECG CSV loaded and visualized successfully. "
                "No heartbeat classification has been performed."
            )

        except Exception as error:
            st.error(
                f"Unable to process this CSV: {error}"
            )

    else:
        st.info(
            "No file uploaded. Showing the simulated ECG signal."
        )

        st.plotly_chart(
            make_ecg_chart(),
            use_container_width=True,
            key="waveform_page_ecg"
        )

        st.caption(
            "The default waveform is simulated, not a real ECG recording."
        )


# -----------------------------
# CLASSIFICATION DEMO PAGE
# -----------------------------
elif page == "Classification Demo":

    st.markdown("### Heartbeat Classification Demo")
    st.warning(
        "Simulation only. Selecting a category manually does not "
        "run the neural network or classify an ECG signal."
    )

    category_info = {
        "N": {
            "name": "Normal",
            "advice": (
                "This demo label is not a medical assessment. "
                "Follow routine healthcare guidance."
            )
        },
        "S": {
            "name": "Supraventricular",
            "advice": (
                "A real result should be reviewed in clinical "
                "context by a qualified healthcare professional."
            )
        },
        "V": {
            "name": "Ventricular",
            "advice": (
                "An actual result suggesting a ventricular pattern "
                "should be assessed promptly by a healthcare professional."
            )
        },
        "F": {
            "name": "Fusion",
            "advice": (
                "An actual result should be reviewed by a qualified "
                "healthcare professional."
            )
        },
        "Q": {
            "name": "Unknown",
            "advice": (
                "An unclear result may require repeat recording "
                "or professional assessment."
            )
        }
    }

    selected = st.selectbox(
        "Choose a demo heartbeat category",
        options=["N", "S", "V", "F", "Q"],
        format_func=lambda key: f"{key} — {category_info[key]['name']}"
    )

    info = category_info[selected]

    st.divider()
    col1, col2 = st.columns(2)

    with col1:
        st.metric("Selected Demo Category", selected)
        st.subheader(info["name"])

    with col2:
        st.subheader("General Next Step")
        st.info(info["advice"])

    st.caption(
        "This page previews the future result interface. "
        "It does not represent a real model prediction."
    )


# -----------------------------
# 9. MODEL EVALUATION PAGE
# -----------------------------
elif page == "Model Evaluation":

    st.markdown("### Neural Network Performance")
    st.caption("Replace these example scores with Member 3's actual test results.")

    m1, m2, m3, m4 = st.columns(4)

    m1.metric("Accuracy", "94%")
    m2.metric("Precision", "91%")
    m3.metric("Recall", "89%")
    m4.metric("F1-score", "90%")

    labels = ["N", "S", "V", "F", "Q"]

    matrix = np.array([
        [90, 2, 3, 1, 4],
        [5, 80, 7, 3, 5],
        [3, 4, 87, 4, 2],
        [4, 3, 5, 84, 4],
        [6, 4, 2, 3, 85]
    ])

    fig = go.Figure(
        data=go.Heatmap(
            z=matrix,
            x=labels,
            y=labels,
            colorscale="Blues",
            colorbar=dict(title="Count"),
            hovertemplate=(
                "Actual: %{y}<br>"
                "Predicted: %{x}<br>"
                "Illustrative count: %{z}<extra></extra>"
            )
        )
    )

    fig.update_layout(
        title="Illustrative Confusion Matrix",
        xaxis_title="Predicted Category",
        yaxis_title="Actual Category",
        height=420,
        paper_bgcolor="white"
    )

    st.plotly_chart(fig, use_container_width=True)

    st.warning(
        "All scores and matrix values are placeholders, not measured "
        "performance of your team's neural network."
    )

# -----------------------------
# 10. CLASSIFICATION HISTORY
# -----------------------------
elif page == "Classification History":

    st.markdown("### Classification History")

    history = pd.DataFrame({
        "Record ID": [
            "DEMO-001",
            "DEMO-002",
            "DEMO-003",
            "DEMO-004",
            "DEMO-005"
        ],
        "Category": ["N", "V", "N", "S", "N"],
        "Model Score": ["96%", "88%", "93%", "85%", "95%"],
        "Source": ["Simulated"] * 5
    })

    st.dataframe(
        history,
        hide_index=True,
        use_container_width=True
    )

    st.download_button(
        "Download demo history as CSV",
        data=history.to_csv(index=False),
        file_name="ecg_demo_history.csv",
        mime="text/csv"
    )

# -----------------------------
# 11. FOOTER
# -----------------------------
st.divider()
st.caption(
    "ECG Heartbeat Classification System | Student prototype | "
    "Simulated data only — not for diagnosis"
)
