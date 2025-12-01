"""
Custom loading spinner with progress tracking for agent execution.
"""

import streamlit as st
from typing import Dict, Any
import time


def show_loading_spinner(message: str = "Processing..."):
    """Display a simple loading spinner."""
    with st.spinner(message):
        pass


def show_agent_progress(progress_data: Dict[str, Any]):
    """
    Display detailed progress for multi-agent execution.
    
    Args:
        progress_data: Dictionary containing stage, message, and progress percentage
    """
    
    stage = progress_data.get('stage', 'processing')
    message = progress_data.get('message', 'Processing...')
    progress = progress_data.get('progress', 0)
    
    # Stage emojis
    stage_emojis = {
        'validation': '✅',
        'intent_parsing': '🧠',
        'workflow_planning': '📊',
        'data_collection': '🌐',
        'personalization': '🎯',
        'budget_optimization': '💰',
        'itinerary_generation': '📋',
        'pdf_generation': '📄',
        'complete': '🎉'
    }
    
    emoji = stage_emojis.get(stage, '⚙️')
    
    # Display progress bar
    st.progress(progress / 100, text=f"{emoji} {message}")
    
    return progress


def show_multi_stage_progress(current_stage: str, all_stages: list):
    """
    Show multi-stage progress with checkmarks for completed stages.
    
    Args:
        current_stage: Currently executing stage
        all_stages: List of all stages in order
    """
    
    stage_status = {
        'validation': 'Validating Request',
        'intent_parsing': 'Analyzing Intent',
        'workflow_planning': 'Planning Workflow',
        'data_collection': 'Fetching Data',
        'personalization': 'Personalizing',
        'budget_optimization': 'Optimizing Budget',
        'itinerary_generation': 'Generating Itinerary',
        'pdf_generation': 'Creating PDF',
        'complete': 'Complete'
    }
    
    for stage in all_stages:
        if stage == current_stage:
            st.markdown(f"⚙️ **{stage_status.get(stage, stage)}** - In Progress")
        elif all_stages.index(stage) < all_stages.index(current_stage):
            st.markdown(f"✅ **{stage_status.get(stage, stage)}** - Complete")
        else:
            st.markdown(f"⏳ **{stage_status.get(stage, stage)}** - Pending")


class ProgressTracker:
    """Class to track and display progress throughout the pipeline."""
    
    def __init__(self, container):
        """Initialize progress tracker with a container."""
        self.container = container
        self.progress_bar = None
        self.status_text = None
        self.stages_completed = []
        
    def initialize(self):
        """Initialize progress display elements."""
        with self.container:
            self.status_text = st.empty()
            self.progress_bar = st.empty()
    
    def update(self, stage: str, message: str, progress: int):
        """Update progress display."""
        if self.status_text and self.progress_bar:
            self.status_text.markdown(f"**{message}**")
            self.progress_bar.progress(progress / 100)
            
            if stage not in self.stages_completed and progress > 0:
                self.stages_completed.append(stage)
    
    def complete(self):
        """Mark progress as complete."""
        if self.status_text and self.progress_bar:
            self.status_text.markdown("**✅ Travel plan ready!**")
            self.progress_bar.progress(100 / 100)


def display_processing_animation():
    """Display an animated processing message."""
    
    messages = [
        "🤖 AI agents are working on your itinerary...",
        "✈️ Searching for the best flights...",
        "🏨 Finding perfect accommodation...",
        "🎯 Personalizing recommendations...",
        "💰 Optimizing your budget...",
        "📋 Creating your itinerary..."
    ]
    
    placeholder = st.empty()
    
    for i, msg in enumerate(messages):
        with placeholder.container():
            st.info(msg)
            time.sleep(0.5)  # Brief pause between messages
    
    placeholder.empty()


def show_stage_cards(stages: list, current_stage: str):
    """
    Display stage progress as cards.
    
    Args:
        stages: List of stage dictionaries with name and status
        current_stage: Currently active stage
    """
    
    cols = st.columns(len(stages))
    
    for i, (col, stage) in enumerate(zip(cols, stages)):
        with col:
            stage_name = stage.get('name', f'Stage {i+1}')
            stage_status = 'complete' if stage.get('complete') else ('active' if stage_name == current_stage else 'pending')
            
            if stage_status == 'complete':
                st.success(f"✅ {stage_name}")
            elif stage_status == 'active':
                st.info(f"⚙️ {stage_name}")
            else:
                st.secondary(f"⏳ {stage_name}")
