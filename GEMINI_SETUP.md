# Gemini setup for ECG Heartbeat Classification Dashboard

## Configure the API key

1. Create a Gemini API key in Google AI Studio: https://aistudio.google.com/app/apikey
2. In Streamlit Community Cloud, open your app's **Settings → Secrets**.
3. Add this TOML entry (replace the placeholder locally in the Streamlit Secrets UI only):

```toml
GEMINI_API_KEY = "your-real-api-key"
# Optional; defaults to gemini-2.5-flash
GEMINI_MODEL = "gemini-2.5-flash"
```

For local development, create `.streamlit/secrets.toml` with the same entries. Do not commit that file or put a real key in source code, screenshots, issues, or chat. If a key is exposed, revoke it and create a new one. The app also accepts `GEMINI_API_KEY` as an environment variable.

## What Gemini receives

After a prediction run, the app calculates a compact JSON summary: counts and percentages for N/S/V/F/Q, mean CNN confidence, number of low-confidence beats, an approximate heart-rate estimate from detected centers, detector metadata, and reference-mode metrics when available. It does **not** send the raw waveform or full beat-by-beat table to Gemini.

The English report can be downloaded as Markdown. The chat assistant answers follow-up questions about the current run and keeps a short in-session conversation history.

## Medical and privacy limitations

- This is a research/student prototype, not a medical device and not clinically validated.
- N/S/V/F/Q are classifier labels, not diagnoses. The LLM must not be used to diagnose, rule out disease, or recommend treatment.
- Heart rate estimates may be wrong if R peaks are missed, extra peaks are detected, or the signal quality is poor.
- Review results with a qualified healthcare professional. If a person reports chest pain, fainting, or severe shortness of breath, seek urgent local medical care.
- Only send data you are authorized to process. This integration sends derived summary metrics and user questions to Google's Gemini API; review Google's current terms and privacy documentation before using real/patient data.
- API calls may incur usage charges and are subject to account quotas. Monitor usage in Google AI Studio.
