
# Barangay Microclimate Forecast Dashboard (v2)

New features:
- **Multi-site comparison** (side-by-side)
- **Auto-tune α, β** per site & signal
- **AQI category coloring**
- **Email/Telegram alerts**
- **Dockerfile** for deployment

## Run
```bash
python -m venv .venv
source .venv/bin/activate  # Windows: .venv\Scriptsctivate
pip install -r requirements.txt
streamlit run app.py
```

## Docker
```bash
docker build -t barangay-forecast .
docker run -it --rm -p 8501:8501 -v $(pwd)/data:/app/data barangay-forecast
```
