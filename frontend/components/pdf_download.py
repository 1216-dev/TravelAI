"""
PDF download component for exporting itineraries.
"""

import streamlit as st
import os
from typing import Optional


def create_download_button(
    pdf_path: str,
    button_text: str = "📥 Download Itinerary (PDF)",
    key: str = "download_pdf"
) -> bool:
    """
    Create a download button for PDF itinerary.
    
    Args:
        pdf_path: Path to the PDF file
        button_text: Text to display on button
        key: Unique key for the button
        
    Returns:
        True if button was clicked
    """
    
    if not os.path.exists(pdf_path):
        st.error("PDF file not found. Please generate the itinerary first.")
        return False
    
    # Read PDF file
    with open(pdf_path, 'rb') as f:
        pdf_bytes = f.read()
    
    # Create download button
    filename = os.path.basename(pdf_path)
    
    clicked = st.download_button(
        label=button_text,
        data=pdf_bytes,
        file_name=filename,
        mime='application/pdf',
        key=key,
        use_container_width=True
    )
    
    if clicked:
        st.success("✅ PDF downloaded successfully!")
    
    return clicked


def create_calendar_download_button(
    calendar_path: str,
    button_text: str = "🗓️ Add to Calendar (.ics)",
    key: str = "download_calendar"
) -> bool:
    """
    Create a download button for calendar events.
    
    Args:
        calendar_path: Path to the .ics file
        button_text: Text to display on button
        key: Unique key for the button
        
    Returns:
        True if button was clicked
    """
    
    if not os.path.exists(calendar_path):
        st.error("Calendar file not found.")
        return False
    
    # Read calendar file
    with open(calendar_path, 'rb') as f:
        cal_bytes = f.read()
    
    # Create download button
    filename = os.path.basename(calendar_path)
    
    clicked = st.download_button(
        label=button_text,
        data=cal_bytes,
        file_name=filename,
        mime='text/calendar',
        key=key,
        use_container_width=True
    )
    
    if clicked:
        st.success("✅ Calendar events downloaded! Import into your calendar app.")
    
    return clicked


def display_export_options(pdf_path: Optional[str], calendar_path: Optional[str]):
    """
    Display all export options in a nice layout.
    
    Args:
        pdf_path: Path to PDF file
        calendar_path: Path to calendar file
    """
    
    st.markdown("### 📤 Export Options")
    
    col1, col2 = st.columns(2)
    
    with col1:
        if pdf_path:
            create_download_button(pdf_path, key="main_pdf_download")
        else:
            st.info("PDF will be available after itinerary generation")
    
    with col2:
        if calendar_path:
            create_calendar_download_button(calendar_path, key="main_calendar_download")
        else:
            st.info("Calendar events will be available after itinerary generation")
    
    # Additional export options
    with st.expander("📋 More Export Options"):
        st.markdown("""
        - **PDF Itinerary**: Complete travel plan with all details
        - **Calendar Events**: Import flights, hotels, and activities to your calendar
        - **JSON Data**: Raw itinerary data (for developers)
        """)
        
        if pdf_path:
            st.caption(f"PDF Size: {os.path.getsize(pdf_path) / 1024:.1f} KB")


def show_share_options():
    """Display sharing options for the itinerary."""
    
    st.markdown("### 🔗 Share Your Itinerary")
    
    col1, col2, col3 = st.columns(3)
    
    with col1:
        if st.button("📧 Email", use_container_width=True):
            st.info("Email sharing feature coming soon!")
    
    with col2:
        if st.button("💬 WhatsApp", use_container_width=True):
            st.info("WhatsApp sharing feature coming soon!")
    
    with col3:
        if st.button("📱 SMS", use_container_width=True):
            st.info("SMS sharing feature coming soon!")
