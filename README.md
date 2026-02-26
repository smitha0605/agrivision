# 🌾 AgriVision — The Farm Brain

> Real agricultural intelligence for Indian farmers. A full-stack ML-powered advisory platform built with FastAPI + Vanilla JS.

---

## What It Does

AgriVision is a smart farming advisory system that goes beyond pretty dashboards. It **knows your farm**, tracks your crops across a season, and gives you actionable decisions — not just information.

### Features

| Feature | What it does |
|---|---|
| 🌿 **Crop Disease Detection** | Upload a photo of a sick leaf → get disease name, severity, and treatment in seconds |
| 🧪 **ICAR Fertilizer Calculator** | Enter your soil test values (N, P, K, pH) → get exact fertilizer doses, stage-wise schedule, and product cost based on Indian Council of Agricultural Research standards |
| 🌤 **7-Day Farm Planner** | Reads live weather → tells you when to irrigate, spray, and apply fertilizer each day |
| 📈 **Mandi Price Forecast** | Historical Indian mandi price trends + 3-week prediction with sell/hold advice |
| 🏡 **Farm Profile + Memory** | App remembers your farm, crop, and sow date — every recommendation is personalised |
| 🤖 **AI Assistant** | Farming advisor with full farm context. Uses OpenAI GPT-4o-mini if key is set, falls back to a rich rule-based engine |

---

## Tech Stack

| Layer | Technology |
|---|---|
| Backend | FastAPI (Python) |
| ML Models | scikit-learn — RandomForest, KNN |
| Database | SQLite (auto-created on first run) |
| Frontend | HTML5 + CSS3 + Vanilla JavaScript (single-page app) |
| Image Processing | Pillow |
| AI Assistant | OpenAI GPT-4o-mini (optional) |
| Deployment | Render.com |

---

## Run Locally

> This section is for other developers who want to run this project on their own machine. If you're using the live deployed version, just visit the URL directly.

### 1. Clone the repo

```bash
git clone https://github.com/smitha0605/agrivision.git
cd agrivision
```

> Replace `smitha0605` with your actual GitHub username after you push the repo.

### 2. Create a virtual environment

```bash
python -m venv venv
source venv/bin/activate        # Mac/Linux
venv\Scripts\activate           # Windows
```

### 3. Install dependencies

```bash
pip install -r requirements.txt
```

### 4. Run the server

```bash
python main.py
```

Open `http://localhost:7860` in your browser.

### 5. AI Assistant (Optional)

The AI assistant works out of the box without any key — it uses a built-in farming knowledge engine. If you want to upgrade it to GPT-4o-mini, set an OpenAI API key (available at platform.openai.com):

```bash
export OPENAI_API_KEY=sk-...   # Mac/Linux
set OPENAI_API_KEY=sk-...      # Windows
```

---

## Deploy to Render.com (Free)

This repo includes a `render.yaml` that auto-configures everything.

1. Push this repo to GitHub
2. Go to [render.com](https://render.com) → **New Web Service**
3. Connect your GitHub repo
4. Render will detect `render.yaml` and configure automatically
5. Click **Deploy** — your app will be live at `https://agrivision.onrender.com`

> **Note:** On the free tier, the server sleeps after 15 minutes of inactivity and takes ~30 seconds to wake up on first visit. Upgrade to Starter ($7/month) to keep it always-on.

---

## Project Structure

```
agrivision/
├── main.py                  # FastAPI backend — all ML, API routes, logic
├── requirements.txt         # Python dependencies
├── render.yaml              # Render.com deployment config
├── Procfile                 # Process file (Render fallback)
├── .gitignore
├── README.md
├── data/
│   ├── crop_recommendation.csv    # 2,200 rows — N, P, K, temp, humidity, pH, rainfall
│   ├── cleaned_fertilizer.csv     # Fertilizer training data
│   └── cleaned_agriculture.csv    # Agriculture dataset
└── static/
    ├── index.html           # Full frontend — all UI, CSS, JS in one file
    ├── css/                 # (available for future CSS separation)
    ├── js/                  # (available for future JS separation)
    └── img/                 # (available for future image assets)
```

---

## ML Models

| Model | Algorithm | Dataset | Notes |
|---|---|---|---|
| Crop Recommendation | RandomForest (300 trees) | 2,200 rows, 22 crops | 5-fold CV accuracy: ~97% |
| Fertilizer Recommendation | RandomForest + ICAR tables | ICAR published standards | Rule-based core with ML layer |
| Disease Detection | Color + texture feature RF | Synthetic signatures | Image-based, 6 condition classes |
| Crop KNN | K-Nearest Neighbours | 2,200 rows | Distance-weighted voting |
| Price Forecast | Exponential smoothing + trend | Historical mandi data | 3-week horizon |

---

## Target Users

- **Agri-extension workers** (Krishi Mitra, KVK) — use as a decision support tool for farmer visits
- **Progressive young farmers** (18–35) with smartphone access
- **Agri startups and NGOs** building advisory services
- **Agricultural students and researchers**

---

## Known Limitations

- **Disease detection** uses image color/texture analysis, not a deep CNN. A production version would use MobileNetV2 trained on the PlantVillage dataset (54,000 images). This version is a working approximation.
- **Price data** is based on historical mandi price patterns. The production upgrade would pull live data from the [data.gov.in Agmarknet API](https://data.gov.in/resource/daily-market-prices-agri-horticultural-commodities).
- **Language**: Currently English-only. Most Indian farmers communicate in regional languages — Hindi, Punjabi, Marathi, etc. A WhatsApp bot interface in Hindi would dramatically improve real-world reach.

---

## API Endpoints

| Method | Endpoint | Description |
|---|---|---|
| POST | `/api/farm` | Create farm profile |
| GET | `/api/farm/{id}` | Get farm + active crops |
| POST | `/api/crop` | Add crop to farm |
| POST | `/api/soil-test` | Record soil test |
| POST | `/api/recommend-crop` | KNN crop recommendation |
| POST | `/api/fertilizer-calc` | ICAR fertilizer calculation |
| POST | `/api/disease-detect` | Leaf image disease detection |
| GET | `/api/price/{crop}` | Price history + forecast |
| POST | `/api/farm-plan` | Weather-based 7-day farm plan |
| POST | `/api/chat` | AI farming assistant |
| GET | `/api/schemes` | Government scheme information |

---

*Solving real problems faced by Indian farmers — from soil to sale.*
