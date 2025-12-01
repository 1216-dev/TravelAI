# 🌍 AI Travel Planner

An intelligent, adaptive travel planning system powered by multi-agent AI orchestration, real-time data integration, and personalized recommendations.
<img width="2132" height="1320" alt="image" src="https://github.com/user-attachments/assets/e007d6aa-c42e-44b0-bfa0-7cf5436ef815" />

<img width="3420" height="2018" alt="image" src="https://github.com/user-attachments/assets/50840af0-3a99-4fbb-81ec-4774c8f4a1df" />

<img width="3420" height="2010" alt="image" src="https://github.com/user-attachments/assets/4b553c21-f83c-405e-9c9d-831627024000" />



## 🧠 Overview

This system transforms fragmented travel planning into an intelligent, dynamic experience using:
<img width="2502" height="1308" alt="image" src="https://github.com/user-attachments/assets/d4905265-cd7a-4cfc-935f-d2ac3c7bea50" />
<img width="1684" height="2208" alt="image" src="https://github.com/user-attachments/assets/5527e222-71b6-4cd2-9147-d6c06628c559" />


- **Llama-based Orchestrator** for reasoning & planning
- **GNN Agent** for personalized recommendations
- **Real-time API Integration** for flights, hotels, and weather
- **Budget Optimizer** for cost vs. comfort tradeoffs
- **Streamlit UI** for dynamic user interaction
- **PDF Export** for downloadable itineraries

## 🏗️ Architecture

```
ai-travel-planner/
├── backend/               # Core AI agents and orchestration
│   ├── orchestrator.py   # Llama model for intent parsing
│   ├── api_manager.py    # Real-time/mock API integration
│   ├── personalization_gnn.py  # GNN-based personalization
│   ├── budget_optimizer.py     # Cost optimization
│   ├── itinerary_agent.py      # Itinerary generation + PDF
│   ├── utils/            # Utilities and configuration
│   └── main.py           # Unified orchestrator entry
│
└── frontend/             # Streamlit web interface
    ├── app.py           # Main Streamlit app
    ├── components/      # UI components
    └── styles/          # Custom CSS and theming
```

## 🚀 Quick Start

### Easy Way (Recommended) 🎯

```bash
# Just run this - it does everything!
./run.sh
```

Then choose option 1 for the Web UI.

### Manual Setup

**Step 1:** Install Dependencies

```bash
pip install -r requirements.txt
```

**Step 2:** Set Up API Keys (REQUIRED for flight and hotel data)

Create a `.env` file:
```bash
# REQUIRED for flights and hotels
SERPAPI_API_KEY=your_key_here  # Get from serpapi.com (100 searches/month free)

# Optional for weather
OPENWEATHER_API_KEY=your_key_here  # Free from openweathermap.org

# Optional for activities
YELP_API_KEY=your_key_here  # Free from yelp.com/developers
```

**Step 3:** Run the Application

```bash
# Web UI (Recommended)
streamlit run frontend/app.py

# Or test backend directly
python backend/main.py
```

### 📚 Detailed Setup Guides

- **`INSTALL.txt`** - Quick visual install guide (start here!)
- **`SETUP.md`** - Comprehensive setup documentation
- **`API_SETUP_GUIDE.md`** - API configuration details
- **`QUICKSTART.md`** - Quick reference guide

### 🧪 Verify Installation

```bash
# Test SERP API integration (flights + hotels)
python test_serpapi.py

# Test other APIs
python tests/test_apis.py

# Test workflow
python tests/test_workflow.py

# Test new features  
python tests/test_new_features.py
```

## ✨ Features

### User Input
- Destination selection
- Date range picker
- Budget slider
- Preference tags (Adventure, Luxury, Nature, etc.)
- Travel group size

### Live Processing
- Real-time agent progress updates
- Multi-agent orchestration visualization
- Animated loading states

### Results Dashboard
- ✈️ **Flight recommendations** with airline logos, real-time pricing, and carbon emissions
- 🏨 **Hotel options** with images, ratings, reviews, and amenities
- 🌤️ **Weather forecasts** for each day
- 🗺️ **Interactive map** integration
- 💰 **Budget breakdown** with detailed cost allocation
- 📥 **PDF itinerary** download
- 🗓️ **Calendar export** (.ics)

### Feedback System
- User satisfaction ratings
- Itinerary accuracy feedback
- Continuous improvement analytics

## 🎨 Design Philosophy

- **Theme**: Deep blue + coral accents
- **Layout**: Responsive grid-based design
- **Animations**: Smooth transitions and loading states
- **UX**: Intuitive, modern, and accessible

## 🧩 Backend Modules

### Orchestrator (`orchestrator.py`)
- Parses user intent using Llama model
- Validates constraints
- Coordinates agent workflow

### API Manager (`api_manager.py`)
- **SERP API Google Flights** - Real-time flight data with airline logos
- **SERP API Google Hotels** - Hotel data with images and ratings
- **OpenWeatherMap** - Weather forecasts
- **Budget-based filtering** - Filters results by budget constraints

### Personalization GNN (`personalization_gnn.py`)
- Graph Neural Network for user preferences
- Historical behavior analysis
- Contextual recommendations

### Budget Optimizer (`budget_optimizer.py`)
- Pareto optimization for cost vs. comfort
- Multiple itinerary options
- Value scoring

### Itinerary Agent (`itinerary_agent.py`)
- Consolidates all recommendations
- Generates PDF itineraries
- Exports calendar events

## 📋 Requirements

- Python 3.9+
- PyTorch
- PyTorch Geometric
- Streamlit
- Transformers (Hugging Face)
- ReportLab
- Other dependencies in `requirements.txt`

## 🔧 Configuration

Edit `backend/utils/config.yaml` to customize:
- API keys and endpoints
- Model configurations
- Budget constraints
- Personalization parameters

## 📝 Testing

Sample input is provided in `backend/utils/sample_input.json` for testing the backend pipeline independently.

## 🤝 Contributing

This is a demonstration project showcasing AI orchestration and real-time travel planning capabilities.

## 📄 License

MIT License


