"""
Feedback form component for collecting user satisfaction data.
"""

import streamlit as st
import json
import os
from datetime import datetime
from typing import Dict, Any


def show_feedback_form(itinerary_id: str) -> bool:
    """
    Display feedback form for the generated itinerary.
    
    Args:
        itinerary_id: ID of the itinerary to collect feedback for
        
    Returns:
        True if feedback was submitted
    """
    
    st.markdown("### 💬 Rate Your Experience")
    st.caption("Help us improve by sharing your feedback")
    
    with st.form("feedback_form"):
        # Overall satisfaction
        satisfaction = st.slider(
            "Overall Satisfaction",
            min_value=1,
            max_value=5,
            value=4,
            help="How satisfied are you with the generated itinerary?"
        )
        
        # Specific ratings
        st.markdown("#### Rate Specific Aspects")
        
        col1, col2 = st.columns(2)
        
        with col1:
            accuracy_rating = st.select_slider(
                "Accuracy",
                options=["Poor", "Fair", "Good", "Very Good", "Excellent"],
                value="Good",
                help="How accurate was the itinerary to your preferences?"
            )
            
            value_rating = st.select_slider(
                "Value for Money",
                options=["Poor", "Fair", "Good", "Very Good", "Excellent"],
                value="Good",
                help="How well optimized was the budget?"
            )
        
        with col2:
            relevance_rating = st.select_slider(
                "Relevance",
                options=["Poor", "Fair", "Good", "Very Good", "Excellent"],
                value="Good",
                help="How relevant were the recommendations?"
            )
            
            usability_rating = st.select_slider(
                "Ease of Use",
                options=["Poor", "Fair", "Good", "Very Good", "Excellent"],
                value="Good",
                help="How easy was it to use the planner?"
            )
        
        # What did you like?
        liked = st.multiselect(
            "What did you like? (Select all that apply)",
            options=[
                "Flight options",
                "Hotel recommendations",
                "Activity suggestions",
                "Budget optimization",
                "Daily schedule",
                "PDF itinerary",
                "User interface",
                "Processing speed"
            ]
        )
        
        # What could be improved?
        improvements = st.text_area(
            "What could be improved?",
            placeholder="Share your suggestions for improvement...",
            height=100
        )
        
        # Would you recommend?
        recommend = st.radio(
            "Would you recommend this planner to others?",
            options=["Yes", "Maybe", "No"],
            horizontal=True
        )
        
        # Additional comments
        comments = st.text_area(
            "Additional Comments (Optional)",
            placeholder="Any other feedback you'd like to share...",
            height=100
        )
        
        # Submit button
        submitted = st.form_submit_button(
            "Submit Feedback",
            use_container_width=True,
            type="primary"
        )
        
        if submitted:
            # Save feedback
            feedback_data = {
                'itinerary_id': itinerary_id,
                'timestamp': datetime.now().isoformat(),
                'ratings': {
                    'overall_satisfaction': satisfaction,
                    'accuracy': accuracy_rating,
                    'value': value_rating,
                    'relevance': relevance_rating,
                    'usability': usability_rating
                },
                'liked': liked,
                'improvements': improvements,
                'recommend': recommend,
                'comments': comments
            }
            
            save_feedback(feedback_data)
            
            st.success("✅ Thank you for your feedback!")
            st.balloons()
            
            return True
    
    return False


def save_feedback(feedback_data: Dict[str, Any]):
    """
    Save feedback data to JSON file.
    
    Args:
        feedback_data: Dictionary containing feedback information
    """
    
    # Create feedback directory
    feedback_dir = "output/feedback"
    os.makedirs(feedback_dir, exist_ok=True)
    
    # Load existing feedback
    feedback_file = os.path.join(feedback_dir, "feedback.json")
    
    if os.path.exists(feedback_file):
        with open(feedback_file, 'r') as f:
            all_feedback = json.load(f)
    else:
        all_feedback = []
    
    # Append new feedback
    all_feedback.append(feedback_data)
    
    # Save updated feedback
    with open(feedback_file, 'w') as f:
        json.dump(all_feedback, f, indent=2)


def show_quick_feedback():
    """Display a quick emoji-based feedback widget."""
    
    st.markdown("### 😊 Quick Feedback")
    
    col1, col2, col3, col4, col5 = st.columns(5)
    
    emojis = ["😞", "😕", "😐", "😊", "😍"]
    labels = ["Poor", "Fair", "Good", "Great", "Excellent"]
    
    for i, (col, emoji, label) in enumerate(zip([col1, col2, col3, col4, col5], emojis, labels)):
        with col:
            if st.button(emoji, key=f"emoji_{i}", use_container_width=True):
                st.success(f"Thanks for rating us {label}!")
                # Save quick feedback
                save_feedback({
                    'type': 'quick_emoji',
                    'rating': i + 1,
                    'label': label,
                    'timestamp': datetime.now().isoformat()
                })


def display_feedback_stats():
    """Display aggregated feedback statistics (for admin/testing)."""
    
    feedback_file = "output/feedback/feedback.json"
    
    if not os.path.exists(feedback_file):
        st.info("No feedback data available yet.")
        return
    
    with open(feedback_file, 'r') as f:
        all_feedback = json.load(f)
    
    if not all_feedback:
        st.info("No feedback data available yet.")
        return
    
    st.markdown("### 📊 Feedback Statistics")
    
    # Calculate average ratings
    total_feedback = len(all_feedback)
    
    # Filter out quick emoji feedback for detailed stats
    detailed_feedback = [f for f in all_feedback if f.get('type') != 'quick_emoji']
    
    if detailed_feedback:
        avg_satisfaction = sum(f['ratings']['overall_satisfaction'] for f in detailed_feedback) / len(detailed_feedback)
        
        col1, col2, col3 = st.columns(3)
        
        with col1:
            st.metric("Total Responses", total_feedback)
        with col2:
            st.metric("Average Rating", f"{avg_satisfaction:.1f}/5")
        with col3:
            recommend_yes = sum(1 for f in detailed_feedback if f.get('recommend') == 'Yes')
            recommend_pct = (recommend_yes / len(detailed_feedback)) * 100
            st.metric("Would Recommend", f"{recommend_pct:.0f}%")
        
        # Most liked features
        all_liked = []
        for f in detailed_feedback:
            all_liked.extend(f.get('liked', []))
        
        if all_liked:
            from collections import Counter
            liked_counts = Counter(all_liked)
            
            st.markdown("#### 👍 Most Appreciated Features")
            for feature, count in liked_counts.most_common(5):
                st.write(f"• {feature}: {count} votes")
