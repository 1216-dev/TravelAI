# 🚀 Travelopedia Setup Guide

A quick guide to get Travelopedia up and running on your machine.

---

## 📋 Prerequisites

- Python 3.8 or higher
- pip (Python package manager)
- Internet connection (for API calls)

---

## ⚡ Quick Start (5 Minutes)

### 1. Clone/Download the Project
```bash
cd /path/to/Travelopedia
```

### 2. Install Dependencies
```bash
pip install -r requirements.txt
```

### 3. Set Up Environment Variables
Create a `.env` file in the root directory:

```bash
# Create .env file
cat > .env << 'EOF'
# Weather API (FREE - Required)
OPENWEATHER_API_KEY=your_openweather_key_here

# Activities APIs (Optional but recommended)
YELP_API_KEY=your_yelp_key_here

# Flight & Hotel APIs (Optional - will use mock data if not provided)
AVIATIONSTACK_API_KEY=your_aviationstack_key_here
RAPIDAPI_KEY=your_rapidapi_key_here
EOF
```

**Get Free API Keys:**
- OpenWeatherMap: https://openweathermap.org/api (FREE, 1,000 calls/day)
- Yelp Fusion: https://www.yelp.com/developers (FREE, 5,000 calls/day)
- Aviationstack: https://aviationstack.com (FREE, 100 calls/month)
- RapidAPI: https://rapidapi.com (FREE tier available)

**Note:** System works with mock data if APIs aren't configured!

### 4. Run the Application

**Option A: Web UI (Recommended)**
```bash
streamlit run frontend/app.py
```
Then open your browser to: http://localhost:8501

**Option B: Command Line Test**
```bash
python backend/main.py
```

---

## 🧪 Verify Installation

### Test APIs
```bash
python tests/test_apis.py
```

Expected output:
```
✅ Weather API Working! (OpenWeatherMap)
✅ Activities API Working! (OpenStreetMap - FREE!)
⚠️  Flights: Using Mock Data
⚠️  Hotels: Using Mock Data
```

### Test Workflow
```bash
python tests/test_workflow.py
```

### Test New Features
```bash
python tests/test_new_features.py
```

---

## 📁 Project Structure

```
Travelopedia/
├── backend/              # Core AI agents and logic
│   ├── main.py          # Main orchestration pipeline
│   ├── orchestrator.py  # Llama-based orchestrator
│   ├── personalization_gnn.py  # GNN personalization
│   ├── budget_optimizer.py     # Budget optimization
│   ├── itinerary_agent.py      # Itinerary generation
│   ├── api_manager.py          # API integrations
│   └── utils/           # Validators, config, logging
├── frontend/            # Streamlit UI
│   ├── app.py          # Main UI application
│   └── components/     # UI components
├── tests/              # Test suites
├── output/             # Generated itineraries & PDFs
├── logs/               # Application logs
└── requirements.txt    # Python dependencies
```

---

## 🎯 Usage Examples

### Using the Web UI

1. **Launch the app:**
   ```bash
   streamlit run frontend/app.py
   ```

2. **Fill in the form:**
   - Destination: Paris
   - Origin: New York
   - Dates: Pick your travel dates
   - Budget: $2500
   - Group Size: 2
   - Preferences: Cultural, Culinary

3. **Click "Plan My Trip"**

4. **Download your itinerary** (PDF + Calendar)

### Using the Command Line

```bash
# Run with sample data
python backend/main.py

# Check output folder
ls -la output/itineraries/
```

---

## 🔧 Configuration

### Edit API Settings
Edit `backend/utils/config.yaml`:

```yaml
apis:
  weather:
    enabled: true
    provider: openweathermap
  
  activities:
    enabled: true
    provider: osm_nominatim  # Free, no key needed!
  
  flight:
    enabled: false  # Set to true when API key added
  
  hotel:
    enabled: false  # Set to true when API key added
```

### Edit Budget Defaults
Edit `backend/utils/config.yaml`:

```yaml
budget:
  default_allocation:
    transport: 0.30      # 30% for flights
    accommodation: 0.35  # 35% for hotels
    activities: 0.20     # 20% for activities
    food: 0.10          # 10% for food
    miscellaneous: 0.05  # 5% for misc
```

---

## 🐛 Troubleshooting

### "ModuleNotFoundError"
```bash
# Reinstall dependencies
pip install -r requirements.txt
```

### "API key not found"
```bash
# Check .env file exists
ls -la .env

# Verify it's loaded
python -c "from dotenv import load_dotenv; load_dotenv(); import os; print('Keys loaded:', bool(os.getenv('OPENWEATHER_API_KEY')))"
```

### "Permission denied" (macOS)
```bash
chmod +x run.sh
./run.sh
```

### Streamlit Port Already in Use
```bash
# Use different port
streamlit run frontend/app.py --server.port 8502
```

### PDF Generation Errors
```bash
# Install reportlab
pip install reportlab
```

---

## 🌟 Features

### ✅ Working Features
- ✅ Real-time weather forecasts (OpenWeatherMap)
- ✅ Activities recommendations (OpenStreetMap + Yelp)
- ✅ GNN-based personalization
- ✅ Budget optimization with alternatives
- ✅ PDF itinerary generation
- ✅ Calendar export (.ics)
- ✅ Multi-comfort level planning
- ✅ Progress tracking UI

### 🔄 Using Mock Data (If APIs Not Configured)
- Flights (intelligent price simulation)
- Hotels (realistic pricing based on location)

### 🚧 Future Enhancements
- Kafka real-time updates
- Automated booking integration
- Email notifications
- Multi-destination planning

---

## 🆘 Getting Help

### Check Documentation
- `README.md` - Project overview
- `ARCHITECTURE_ANALYSIS.md` - Detailed system analysis
- `API_SETUP_GUIDE.md` - API configuration guide
- `QUICKSTART.md` - Quick reference

### Run Tests
```bash
# Test everything
python tests/test_apis.py
python tests/test_workflow.py
python tests/test_new_features.py
```

### Check Logs
```bash
# View recent logs
tail -f logs/travel_planner.log
```

---

## 💡 Tips

1. **Start Simple**: Run without API keys first (uses mock data)
2. **Add APIs Gradually**: Start with Weather (free & easy)
3. **Test Often**: Run `test_apis.py` after adding each key
4. **Check Logs**: Look in `logs/` folder for detailed debugging
5. **Use Mock Data**: Perfect for development and demos

---

## 🚀 Quick Commands Cheat Sheet

```bash
# Install
pip install -r requirements.txt

# Run Web UI
streamlit run frontend/app.py

# Run CLI Test
python backend/main.py

# Test APIs
python tests/test_apis.py

# Test Workflow
python tests/test_workflow.py

# View Logs
tail -f logs/travel_planner.log

# Clean Output
rm -rf output/itineraries/*
```

---

## 📊 System Requirements

**Minimum:**
- Python 3.8+
- 2GB RAM
- 500MB disk space
- Internet connection

**Recommended:**
- Python 3.10+
- 4GB RAM
- 1GB disk space
- Stable internet connection

---

## 🎓 Example API Keys Setup

```bash
# 1. Get OpenWeatherMap key (5 minutes)
# Visit: https://openweathermap.org/api
# Sign up → Get API key → Copy

# 2. Add to .env
echo "OPENWEATHER_API_KEY=abc123def456" >> .env

# 3. Test it
python tests/test_apis.py

# 4. See real weather data! ☀️
```

---

## ✅ Success Checklist

- [ ] Python 3.8+ installed
- [ ] Dependencies installed (`pip install -r requirements.txt`)
- [ ] `.env` file created (at minimum with OPENWEATHER_API_KEY)
- [ ] Tests pass (`python tests/test_apis.py`)
- [ ] Web UI launches (`streamlit run frontend/app.py`)
- [ ] Can generate itinerary
- [ ] Can download PDF

---

## 🎉 You're Ready!

If you see this, you're all set:
```
✅ Weather API Working!
✅ Activities API Working!
⚠️  Using mock data for flights/hotels (optional)

Visit: http://localhost:8501
```

**Happy Travel Planning! ✈️🌍**

---

**Last Updated:** October 19, 2025  
**Version:** 1.0  
**Status:** Production Ready (Beta)
