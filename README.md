# SIH 2026 Crop Health AI Dashboard

Deployment version prepared from the working Colab prototype.

## Included
- EfficientNet-B0 crop disease detection
- YOLO11n pest detection
- Automatic browser location
- Open-Meteo current weather + 3-day forecast
- Weather and overall crop risk
- Multilingual advisory
- IPM + safe-input recommendation
- Expert/lab referral
- Farmer report storage
- Agricultural officer dashboard
- DBSCAN hotspot map
- Follow-up monitoring and feedback
- Field confirmation / AI feedback
- Follow-up history

## Required model files
Place these two files in the same folder as `app.py`:

- `plant_disease_efficientnet_b0.pth`
- `best (1).pt`

These are not duplicated in this deployment package because they were already backed up separately.

## Run locally
```bash
pip install -r requirements.txt
python app.py
```

The app listens on port 7860, or the `PORT` environment variable when supplied by the hosting service.

## Important
The current prototype stores reports/follow-ups/feedback in CSV files. For production deployment, persistent database storage should eventually replace local CSV storage.
