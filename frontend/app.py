"""
Main Streamlit application for AI Travel Planner.
High-fidelity UI for interactive travel planning.
"""

import streamlit as st
import asyncio
import sys
import os
from datetime import datetime, timedelta
from pathlib import Path

# Add backend to path
sys.path.append(str(Path(__file__).parent.parent))

from backend.main import TravelPlannerPipeline
from frontend.components.itinerary_display import (
    display_itinerary_card,
    display_alternative_plans
)
from frontend.components.loading_spinner import ProgressTracker
from frontend.components.pdf_download import display_export_options, show_share_options
from frontend.components.feedback_form import show_feedback_form, show_quick_feedback


# Page configuration
st.set_page_config(
    page_title="AI Travel Planner",
    page_icon="✈️",
    layout="wide",
    initial_sidebar_state="expanded"
)

# Load custom CSS
def load_css():
    """Load custom CSS styling."""
    css_file = Path(__file__).parent / "styles" / "custom.css"
    if css_file.exists():
        with open(css_file) as f:
            st.markdown(f"<style>{f.read()}</style>", unsafe_allow_html=True)

load_css()


def initialize_session_state():
    """Initialize session state variables."""
    if 'itinerary' not in st.session_state:
        st.session_state.itinerary = None
    if 'result' not in st.session_state:
        st.session_state.result = None
    if 'processing' not in st.session_state:
        st.session_state.processing = False


def main():
    """Main application function."""
    
    initialize_session_state()
    
    # Header
    st.markdown("""
        <div class="header">
            <h1>✈️ AI Travel Planner</h1>
            <p>Intelligent, adaptive travel planning powered by multi-agent AI</p>
        </div>
    """, unsafe_allow_html=True)
    
    # Sidebar for input
    with st.sidebar:
        st.markdown("## 🧭 Plan Your Trip")
        
        with st.form("travel_form"):
            # Destination
            destination = st.text_input(
                "Destination",
                value="Los Angeles, USA",
                help="Where do you want to go? You can use city names (e.g., 'New York, USA') or airport codes (e.g., 'JFK')"
            )
            
            origin = st.text_input(
                "Origin",
                value="New York, USA",
                help="Where are you traveling from? You can use city names or airport codes (e.g., 'JFK', 'LAX', 'DEL')"
            )
            
            # Dates
            st.markdown("### 📅 Travel Dates")
            col1, col2 = st.columns(2)
            
            min_date = datetime.now() + timedelta(days=1)
            max_date = datetime.now() + timedelta(days=365)
            
            with col1:
                start_date = st.date_input(
                    "Start Date",
                    value=min_date + timedelta(days=30),
                    min_value=min_date,
                    max_value=max_date
                )
            
            with col2:
                end_date = st.date_input(
                    "End Date",
                    value=min_date + timedelta(days=35),
                    min_value=start_date,
                    max_value=max_date
                )
            
            # Budget
            st.markdown("### 💰 Budget")
            budget = st.slider(
                "Total Budget (USD)",
                min_value=500,
                max_value=20000,
                value=3500,
                step=100,
                help="Your total budget for the trip. For international flights, consider $3000+"
            )
            
            # Show budget recommendation
            if budget < 2000:
                st.info("💡 Tip: International flights typically start at $1500-2000. Consider increasing budget for more options.")
            elif budget < 3000:
                st.info("💡 Tip: For comfort or luxury travel, consider budgets of $3000+")
            
            # Group size
            group_size = st.number_input(
                "Number of Travelers",
                min_value=1,
                max_value=10,
                value=2,
                help="How many people are traveling?"
            )
            
            # Preferences
            st.markdown("### 🎯 Preferences")
            
            preferences = st.multiselect(
                "Trip Style",
                options=[
                    "adventure",
                    "luxury",
                    "nature",
                    "cultural",
                    "culinary",
                    "relaxation",
                    "family_friendly",
                    "romantic"
                ],
                default=["adventure", "nature"],
                help="Select your travel preferences"
            )
            
            comfort_level = st.select_slider(
                "Comfort Level",
                options=["budget", "standard", "comfort", "luxury"],
                value="standard",
                help="Choose your preferred comfort level"
            )
            
            # Hotel preferences
            min_hotel_rating = st.slider(
                "Minimum Hotel Rating",
                min_value=1.0,
                max_value=5.0,
                value=3.5,
                step=0.5,
                help="Minimum acceptable hotel rating"
            )
            
            # Advanced options
            with st.expander("⚙️ Advanced Options"):
                flexible_dates = st.checkbox("Flexible dates", value=True)
                include_activities = st.checkbox("Include activities", value=True)
                max_flight_duration = st.number_input(
                    "Max flight duration (hours)",
                    min_value=1,
                    max_value=30,
                    value=20
                )
            
            # Submit button
            submitted = st.form_submit_button(
                "🚀 Generate Itinerary",
                use_container_width=True,
                type="primary"
            )
            
            if submitted:
                # Validate inputs
                if not destination:
                    st.error("Please enter a destination")
                elif start_date >= end_date:
                    st.error("End date must be after start date")
                else:
                    # Build request
                    request = {
                        'user_id': 'streamlit_user',
                        'query': f"Plan a trip to {destination}",
                        'destination': destination,
                        'origin': origin,
                        'dates': {
                            'start': start_date.strftime("%Y-%m-%d"),
                            'end': end_date.strftime("%Y-%m-%d")
                        },
                        'budget': {
                            'total': float(budget),
                            'currency': 'USD'
                        },
                        'group_size': group_size,
                        'preferences': {
                            'categories': preferences,
                            'comfort_level': comfort_level,
                            'accommodation_type': 'hotel',
                            'dietary_restrictions': [],
                            'special_requirements': []
                        },
                        'constraints': {
                            'max_flight_duration': max_flight_duration,
                            'preferred_airlines': [],
                            'hotel_rating_min': min_hotel_rating,
                            'flexible_dates': flexible_dates
                        },
                        'user_history': {
                            'previous_trips': []
                        }
                    }
                    
                    # Process request
                    process_travel_request(request)
    
    # Main content area
    if st.session_state.processing:
        show_processing_view()
    elif st.session_state.itinerary:
        show_results_view()
    else:
        show_welcome_view()


def show_welcome_view():
    """Display welcome screen with instructions."""
    
    st.markdown("""
        <div class="welcome-section">
            <h2>🌍 Welcome to AI Travel Planner</h2>
            <p>Plan your perfect trip with AI-powered recommendations</p>
        </div>
    """, unsafe_allow_html=True)
    
    col1, col2, col3 = st.columns(3)
    
    with col1:
        st.markdown("""
            ### 🤖 AI-Powered
            Multi-agent system with intelligent orchestration
            - Llama reasoning engine
            - GNN personalization
            - Real-time optimization
        """)
    
    with col2:
        st.markdown("""
            ### ⚡ Real-Time Data
            Live integration with travel APIs
            - Flight comparisons
            - Hotel availability
            - Weather forecasts
        """)
    
    with col3:
        st.markdown("""
            ### 🎯 Personalized
            Tailored to your preferences
            - Budget optimization
            - Activity recommendations
            - Daily schedules
        """)
    
    st.markdown("---")
    
    st.markdown("## 🚀 How It Works")
    
    steps_col1, steps_col2, steps_col3, steps_col4 = st.columns(4)
    
    with steps_col1:
        st.info("**1. Input** 📝\n\nEnter your destination, dates, budget, and preferences")
    
    with steps_col2:
        st.info("**2. Process** ⚙️\n\nAI agents analyze and optimize your travel plan")
    
    with steps_col3:
        st.info("**3. Review** 👀\n\nView personalized itinerary with all details")
    
    with steps_col4:
        st.info("**4. Export** 📥\n\nDownload PDF and add to calendar")
    
    st.markdown("---")
    
    st.markdown("### 👈 Get started by filling out the form in the sidebar!")


def show_processing_view():
    """Display processing view with progress updates."""
    
    st.markdown("## 🔄 Processing Your Request")
    
    # Progress container
    progress_container = st.container()
    
    with progress_container:
        st.info("✨ Our AI agents are working on your perfect itinerary...")
        
        # Show processing stages
        stages = [
            "Validating Request",
            "Analyzing Intent",
            "Fetching Travel Data",
            "Personalizing Recommendations",
            "Optimizing Budget",
            "Generating Itinerary"
        ]
        
        for stage in stages:
            st.markdown(f"⏳ {stage}...")


def show_results_view():
    """Display results view with itinerary."""
    
    result = st.session_state.result
    itinerary = st.session_state.itinerary
    
    # Display itinerary
    display_itinerary_card(itinerary)
    
    st.markdown("---")
    
    # Alternative plans
    if result.get('alternatives'):
        display_alternative_plans(result['alternatives'])
        st.markdown("---")
    
    # Export options
    pdf_path = result.get('pdf_path')
    calendar_path = result.get('calendar_path')
    display_export_options(pdf_path, calendar_path)
    
    st.markdown("---")
    
    # Share options
    show_share_options()
    
    st.markdown("---")
    
    # Feedback
    with st.expander("💬 Share Your Feedback", expanded=False):
        show_feedback_form(itinerary['itinerary_id'])
    
    # Quick feedback
    show_quick_feedback()
    
    # New search button
    st.markdown("---")
    if st.button("🔄 Plan Another Trip", use_container_width=True, type="primary"):
        st.session_state.itinerary = None
        st.session_state.result = None
        st.rerun()


def process_travel_request(request):
    """Process travel request asynchronously."""
    
    st.session_state.processing = True
    
    try:
        # Create progress display
        progress_placeholder = st.empty()
        
        with progress_placeholder.container():
            st.markdown("## 🔄 Processing Your Request")
            progress_bar = st.progress(0)
            status_text = st.empty()
            
            # Define progress callback
            def update_progress(progress_data):
                progress = progress_data.get('progress', 0)
                message = progress_data.get('message', 'Processing...')
                
                progress_bar.progress(progress / 100)
                status_text.markdown(f"**{message}**")
            
            # Process request
            pipeline = TravelPlannerPipeline()
            
            # Run async pipeline
            import asyncio
            result = asyncio.run(pipeline.process_request(request, update_progress))
            
            if result['success']:
                st.session_state.result = result['data']
                st.session_state.itinerary = result['data']['itinerary']
                st.session_state.processing = False
                
                # Clear progress and show success
                progress_placeholder.empty()
                st.success("✅ Your travel itinerary is ready!")
                st.balloons()
                st.rerun()
            else:
                st.session_state.processing = False
                progress_placeholder.empty()
                st.error(f"❌ Error: {', '.join(result['errors'])}")
                if result.get('warnings'):
                    st.warning(f"⚠️ Warnings: {', '.join(result['warnings'])}")
    
    except Exception as e:
        st.session_state.processing = False
        st.error(f"❌ An error occurred: {str(e)}")


if __name__ == "__main__":
    main()
