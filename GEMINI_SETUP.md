# Groq setup for ECG Heartbeat Classification Dashboard

## Configure the API key

1. Create a key in the Groq Console: https://console.groq.com/keys
2. Open your app in Streamlit Community Cloud and go to **Settings → Secrets**.
3. Add the following TOML (paste your real key only in Streamlit Secrets; never in source code):

```toml
GROQ_API_KEY = "your-real-api-key"
# Optional. Use a model ID currently available to your Groq account.
GROQ_MODEL = "llama-3.3-70b-versatile"
```

For local development, put the same entries in `.streamlit/secrets.toml` and do not commit that file. The app also accepts `GROQ_API_KEY` and `GROQ_MODEL` environment variables. Never paste your API key into chat, GitHub, screenshots, or issue reports. If exposed, revoke it and create a replacement.

## What is sent

The app sends a compact JSON summary of model outputs (class counts and percentages, mean confidence, low-confidence count, estimated heart rate, detector metadata, and reference-mode metrics if available) plus your question and a short chat history. It does not send the raw ECG waveform or full beat table to Groq.

If the configured model is unavailable, change `GROQ_MODEL` in Streamlit Secrets to a model ID listed in your Groq Console. Free usage limits and available models can change; check the console for current limits.

## Safety and privacy

- Research/student prototype only; not a medical device and not clinically validated.
- N/S/V/F/Q are classifier labels, not diagnoses. Do not use this assistant to diagnose, rule out disease, or choose treatment.
- Heart-rate estimates may be inaccurate if peaks are missed or extra peaks are detected.
- Review results with a qualified healthcare professional. For chest pain, fainting, or severe shortness of breath, seek urgent local medical care.
- Only submit data you are authorized to process. Review Groq's current terms and privacy documentation before processing real/patient data.
