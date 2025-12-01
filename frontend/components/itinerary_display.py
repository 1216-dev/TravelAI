"""
Itinerary display component for showing travel plans in an elegant card layout.
"""

import streamlit as st
from typing import Dict, Any, List


def display_itinerary_card(itinerary: Dict[str, Any]):
    """Display complete itinerary in card format."""
    
    summary = itinerary['trip_summary']
    dates = summary.get('dates', {})
    st.markdown(f"""
        <div class="itinerary-header">
            <div style="display:flex; justify-content:space-between; align-items:center; gap:1rem; flex-wrap:wrap;">
                <div>
                    <h1>✈️ Your Travel Itinerary</h1>
                    <div style="color:#cbd5f5; margin-top:6px;">{summary.get('origin','')} → {summary['destination']} &nbsp;•&nbsp; {dates.get('start','')} to {dates.get('end','')}</div>
                </div>
                <div style="display:flex; gap:0.5rem; flex-wrap:wrap;">
                    <span class="pill">Group: {summary.get('group_size',1)}</span>
                    <span class="pill">Days: {summary.get('duration_days','')}</span>
                    <span class="pill">Value Score: {itinerary.get('value_score',0)}/100</span>
                </div>
            </div>
        </div>
    """, unsafe_allow_html=True)
    
    # Trip Summary Card
    with st.container():
        st.markdown('<div class="section-card">', unsafe_allow_html=True)
        st.markdown("### 📍 Trip Overview")
        col1, col2, col3, col4 = st.columns(4)
        
        with col1:
            st.metric("Destination", summary['destination'])
        with col2:
            st.metric("Duration", f"{summary['duration_days']} days")
        with col3:
            st.metric("Travelers", summary['group_size'])
        with col4:
            st.metric("Value Score", f"{itinerary.get('value_score', 0)}/100")
        st.markdown('</div>', unsafe_allow_html=True)
    
    st.markdown("---")
    
    # Budget Summary Card
    st.markdown('<div class="section-card">', unsafe_allow_html=True)
    st.markdown("### 💰 Budget Breakdown")
    budget = itinerary['budget_summary']
    
    col1, col2, col3 = st.columns(3)
    with col1:
        st.metric("Total Budget", f"${budget['total_budget']:,.2f}")
    with col2:
        st.metric("Total Cost", f"${budget['total_cost']:,.2f}")
    with col3:
        balance_delta = budget['balance']
        st.metric(
            "Balance",
            f"${abs(balance_delta):,.2f}",
            delta=f"${balance_delta:,.2f}",
            delta_color="normal"
        )
    
    # Detailed breakdown
    with st.expander("📊 View Detailed Breakdown"):
        breakdown = budget['breakdown']
        for category, amount in breakdown.items():
            st.write(f"**{category.replace('_', ' ').title()}:** ${amount:,.2f}")
    st.markdown('</div>', unsafe_allow_html=True)
    
    st.markdown("---")
    
    # Flight Information
    st.markdown('<div class="section-card">', unsafe_allow_html=True)
    st.markdown("### ✈️ Flights")
    if itinerary['transportation']['outbound_flight']:
        # The complete flight object is in outbound_flight (includes both legs)
        flight_data = itinerary['transportation']['outbound_flight']
        display_flight_card(flight_data, "round-trip")
    else:
        # No flights available - show helpful message
        st.markdown("""
        <div class="alert-card warning">
            <div class="alert-title">No live flights found</div>
            <p>We couldn't fetch flights from the live API just now. You can still explore mock options or tweak your search.</p>
            <ul>
                <li>Try shifting your dates or origin airport</li>
                <li>Increase the flight portion of your budget</li>
                <li>Double-check SERP API connectivity or key</li>
            </ul>
            <div class="alert-tip">Tip: International routes often clear at $1500–$3000 for round trips.</div>
        </div>
        """, unsafe_allow_html=True)
    st.markdown('</div>', unsafe_allow_html=True)
    
    st.markdown("---")
    
    # Accommodation
    st.markdown('<div class="section-card">', unsafe_allow_html=True)
    if itinerary['accommodation']:
        st.markdown("### 🏨 Accommodation")
        display_hotel_card(itinerary['accommodation'], itinerary['trip_summary'].get('destination', ''))
    st.markdown('</div>', unsafe_allow_html=True)
    
    st.markdown("---")
    
    # Activities
    st.markdown('<div class="section-card">', unsafe_allow_html=True)
    if itinerary['activities']:
        st.markdown("### 🎯 Recommended Activities")
        display_activities_grid(itinerary['activities'])
    st.markdown('</div>', unsafe_allow_html=True)
    
    st.markdown("---")
    
    # Daily Schedule
    st.markdown('<div class="section-card">', unsafe_allow_html=True)
    st.markdown("### 📅 Daily Schedule")
    display_daily_schedule(itinerary['daily_schedule'])
    st.markdown('</div>', unsafe_allow_html=True)


def display_flight_card(flight: Dict[str, Any], flight_type: str):
    """Display flight information card - shows ONLY actual SERP API data."""
    
    # Debug: Show what data we have
    import json
    with st.expander("🔍 Debug: Flight Data Structure"):
        st.code(json.dumps(flight, indent=2, default=str), language='json')
    
    # Helpers for airline logos
    airline_logos = {
        "delta air lines": "https://upload.wikimedia.org/wikipedia/commons/thumb/6/69/Delta_Air_Lines_Logo.svg/256px-Delta_Air_Lines_Logo.svg.png",
        "united airlines": "https://upload.wikimedia.org/wikipedia/commons/thumb/d/d1/United_Airlines_Logo.svg/256px-United_Airlines_Logo.svg.png",
        "american airlines": "https://upload.wikimedia.org/wikipedia/commons/thumb/6/6f/American_Airlines_logo_2013.svg/256px-American_Airlines_logo_2013.svg.png",
        "air canada": "https://upload.wikimedia.org/wikipedia/commons/thumb/3/33/Air_Canada_Logo.svg/256px-Air_Canada_Logo.svg.png",
    }
    fallback_logo = "https://img.icons8.com/color/96/airplane-take-off.png"
    
    # Display overall flight info at top (from SERP API)
    total_price = flight.get('total_price', 0)
    travel_class = flight.get('travel_class', 'N/A')
    carbon = flight.get('carbon_emissions', 0)
    source = flight.get('data_source', 'live')
    badge = "Live API" if source != "mock" else "Mock Sample"
    fallback_reason = flight.get('fallback_reason')
    
    if total_price > 0:
        st.markdown(f"""
        <div class="flight-summary">
            <div class="summary-left">
                <div class="badge {'badge-live' if source != 'mock' else 'badge-mock'}">{badge}</div>
                <h3>💰 ${total_price:,.0f} &nbsp;|&nbsp; ✈️ {travel_class} &nbsp;|&nbsp; 🌍 {carbon:,} kg CO₂</h3>
            </div>
            <div class="summary-right">
                <span class="pill">Round Trip</span>
            </div>
        </div>
        """, unsafe_allow_html=True)
        if fallback_reason and source == "mock":
            st.caption("Using mock flights while live data is unavailable.")
    
    # Get outbound and return data from SERP API
    outbound = flight.get('outbound', {})
    return_flight = flight.get('return', {})
    
    # Debug info
    st.info(f"📊 Flight data check: Outbound={bool(outbound)}, Return={bool(return_flight)}")
    
    if outbound and return_flight:
        # NEW FORMAT: Separate outbound and return flights
        
        # OUTBOUND FLIGHT
        st.markdown("### 🛫 Outbound Flight")
        outbound_date = outbound.get('date', 'N/A')
        st.markdown(f"<div class='pill'>📅 {outbound_date}</div>", unsafe_allow_html=True)
        
        # Display outbound airline logo
        outbound_airline = outbound.get('airline', 'N/A')
        outbound_logo = outbound.get('airline_logo') or airline_logos.get(outbound_airline.lower(), fallback_logo)
        st.image(outbound_logo, width=110, caption=outbound_airline)
        
        col1, col2 = st.columns(2)
        
        with col1:
            st.markdown(f"""
            <div class="flight-card flight-glow">
                <p><strong>Flight:</strong> {outbound.get('flight_number', 'N/A')}</p>
                <p><strong>Duration:</strong> {outbound.get('duration_hours', 'N/A')} hours</p>
                <p><strong>Stops:</strong> {outbound.get('stops', 0)}</p>
                <hr>
                <div class="leg-row">
                    <div>
                        <div class="mini-pill">Departure</div>
                        <p><strong>{outbound.get('departure', {}).get('airport', 'N/A')}</strong></p>
                        <p>{outbound.get('departure', {}).get('time', 'N/A')}</p>
                    </div>
                    <div class="leg-divider"></div>
                    <div>
                        <div class="mini-pill">Arrival</div>
                        <p><strong>{outbound.get('arrival', {}).get('airport', 'N/A')}</strong></p>
                        <p>{outbound.get('arrival', {}).get('time', 'N/A')}</p>
                    </div>
                </div>
            </div>
            """, unsafe_allow_html=True)
        
        with col2:
            st.markdown(f"""
            <div class="flight-card flight-glow">
                <h4>🧳 Travel Notes</h4>
                <p><strong>Class:</strong> {flight.get('travel_class', 'N/A')}</p>
                <p><strong>Baggage:</strong> {flight.get('baggage_allowance', 'Check airline')}</p>
                <p><strong>Layovers:</strong> {', '.join(outbound.get('layovers', [])) or 'Non-stop'}</p>
            </div>
            """, unsafe_allow_html=True)
        
        # Show outbound layovers if any
        if outbound.get('layovers'):
            with st.expander("🔄 Outbound Layover Details"):
                for i, layover in enumerate(outbound['layovers'], 1):
                    st.write(f"**Stop {i}:** {layover}")
        
        st.markdown("---")
        
        # RETURN FLIGHT
        st.markdown("### 🛬 Return Flight")
        return_date = return_flight.get('date', 'N/A')
        st.markdown(f"<div class='pill'>📅 {return_date}</div>", unsafe_allow_html=True)
        
        # Display return airline logo
        return_airline = return_flight.get('airline', 'N/A')
        return_logo = return_flight.get('airline_logo') or airline_logos.get(return_airline.lower(), fallback_logo)
        st.image(return_logo, width=110, caption=return_airline)
        
        col1, col2 = st.columns(2)
        
        with col1:
            st.markdown(f"""
            <div class="flight-card flight-glow">
                <p><strong>Flight:</strong> {return_flight.get('flight_number', 'N/A')}</p>
                <p><strong>Duration:</strong> {return_flight.get('duration_hours', 'N/A')} hours</p>
                <p><strong>Stops:</strong> {return_flight.get('stops', 0)}</p>
                <hr>
                <div class="leg-row">
                    <div>
                        <div class="mini-pill">Departure</div>
                        <p><strong>{return_flight.get('departure', {}).get('airport', 'N/A')}</strong></p>
                        <p>{return_flight.get('departure', {}).get('time', 'N/A')}</p>
                    </div>
                    <div class="leg-divider"></div>
                    <div>
                        <div class="mini-pill">Arrival</div>
                        <p><strong>{return_flight.get('arrival', {}).get('airport', 'N/A')}</strong></p>
                        <p>{return_flight.get('arrival', {}).get('time', 'N/A')}</p>
                    </div>
                </div>
            </div>
            """, unsafe_allow_html=True)
        
        with col2:
            st.markdown(f"""
            <div class="flight-card flight-glow">
                <h4>🧳 Travel Notes</h4>
                <p><strong>Class:</strong> {flight.get('travel_class', 'N/A')}</p>
                <p><strong>Baggage:</strong> {flight.get('baggage_allowance', 'Check airline')}</p>
                <p><strong>Layovers:</strong> {', '.join(return_flight.get('layovers', [])) or 'Non-stop'}</p>
            </div>
            """, unsafe_allow_html=True)
        
        # Show return layovers if any
        if return_flight.get('layovers') and len(return_flight['layovers']) > 0:
            with st.expander("🔄 Return Layover Details"):
                for i, layover in enumerate(return_flight['layovers'], 1):
                    st.write(f"**Stop {i}:** {layover}")
    
    elif outbound and not return_flight:
        # Only outbound data available
        st.error("❌ **MISSING RETURN FLIGHT DATA!**")
        st.warning("""
        The flight object has outbound data but is missing return flight information.
        This is a backend data issue - the return flight should be populated by the API.
        """)
        
        # Still show outbound
        st.markdown("### 🛫 Outbound Flight (Available)")
        st.markdown(f"**📅 Date:** {outbound.get('date', 'N/A')}")
        
        outbound_logo = outbound.get('airline_logo', '')
        outbound_airline = outbound.get('airline', 'N/A')
        
        if outbound_logo:
            st.image(outbound_logo, width=100, caption=outbound_airline)
        else:
            st.markdown(f"**✈️ Airline:** {outbound_airline}")
        
        col1, col2 = st.columns(2)
        
        with col1:
            st.markdown(f"""
            <div class="flight-card">
                <p><strong>Flight:</strong> {outbound.get('flight_number', 'N/A')}</p>
                <p><strong>Duration:</strong> {outbound.get('duration_hours', 'N/A')} hours</p>
                <p><strong>Stops:</strong> {outbound.get('stops', 0)}</p>
                <hr>
                <h4>🛫 Departure</h4>
                <p><strong>Airport:</strong> {outbound.get('departure', {}).get('airport', 'N/A')}</p>
                <p><strong>Time:</strong> {outbound.get('departure', {}).get('time', 'N/A')}</p>
            </div>
            """, unsafe_allow_html=True)
        
        with col2:
            st.markdown(f"""
            <div class="flight-card">
                <h4>🛬 Arrival</h4>
                <p><strong>Airport:</strong> {outbound.get('arrival', {}).get('airport', 'N/A')}</p>
                <p><strong>Time:</strong> {outbound.get('arrival', {}).get('time', 'N/A')}</p>
            </div>
            """, unsafe_allow_html=True)
    
    else:
        # No valid SERP API flight data
        st.warning("⚠️ No flight information available from SERP API. Please check your API configuration.")


def display_hotel_card(hotel: Dict[str, Any], destination: str = ""):
    """Display hotel information card with images."""
    
    # Display hotel images if available
    images = hotel.get('images', [])
    thumbnail = hotel.get('thumbnail', '')
    source = hotel.get('data_source', 'live')
    badge = "Live API" if source != "mock" else "Mock Sample"
    badge_class = "badge-live" if source != "mock" else "badge-mock"
    
    if images or thumbnail:
        # Show first image or thumbnail
        image_to_show = images[0] if images else thumbnail
        if image_to_show:
            st.image(image_to_show, caption=hotel['name'], use_container_width=True)
    
    col1, col2 = st.columns([2, 1])
    
    with col1:
        st.markdown(f"<div class='badge {badge_class}'>{badge}</div>", unsafe_allow_html=True)
        st.markdown(f"### {hotel['name']}")
        review_count = hotel.get('review_count') or hotel.get('reviews', 0)
        if review_count:
            st.markdown(f"**⭐ Rating:** {hotel['rating']}/5.0 ({int(review_count)} reviews)")
        else:
            st.markdown(f"**⭐ Rating:** {hotel['rating']}/5.0")
        st.markdown(f"**🏨 Type:** {hotel.get('type', 'Hotel')}")
        st.markdown(f"**📍 Location:** {hotel['location']['address']}")
        # Add Google Maps search link
        query = f"{hotel['name']} {destination}".strip().replace(' ', '+')
        st.markdown(f"[🌐 Open in Google Maps](https://www.google.com/maps/search/{query})")
        st.markdown(f"**🛏️ Room Type:** {hotel['room_type']}")
        
        # Show hotel class if available
        if hotel.get('hotel_class'):
            st.markdown(f"**⭐ Class:** {hotel['hotel_class']}")
        
        # Show eco certification
        if hotel.get('eco_certified'):
            st.markdown("**🌿 Eco-Certified**")
        
        if hotel.get('amenities'):
            st.markdown("**🎁 Amenities:**")
            # Display amenities in columns
            amenity_cols = st.columns(3)
            for i, amenity in enumerate(hotel['amenities'][:12]):
                with amenity_cols[i % 3]:
                    st.write(f"• {amenity}")
    
    with col2:
        st.markdown("### 💵 Pricing")
        st.metric("Per Night", f"${hotel['price']['per_night']:,.2f}")
        st.metric("Total", f"${hotel['price']['total']:,.2f}")
        st.caption(f"For {hotel['price'].get('nights', 'N/A')} nights")
        
        st.markdown("### 📋 Policies")
        policies = hotel.get('policies', {})
        st.write(f"**Check-in:** {policies.get('check_in', 'N/A')}")
        st.write(f"**Check-out:** {policies.get('check_out', 'N/A')}")
        
        # Booking link
        if hotel.get('booking_url'):
            st.link_button("📖 View on Google", hotel['booking_url'])
    
    # Show more images if available
    if len(images) > 1:
        with st.expander("📸 View More Photos"):
            img_cols = st.columns(3)
            for i, img in enumerate(images[1:6]):  # Show up to 5 more images
                with img_cols[i % 3]:
                    st.image(img, use_container_width=True)


def display_activities_grid(activities: List[Dict[str, Any]]):
    """Display activities in a grid layout."""
    
    # Show activities in rows of 2
    for i in range(0, len(activities), 2):
        col1, col2 = st.columns(2)
        
        with col1:
            if i < len(activities):
                display_activity_card(activities[i], i + 1)
        
        with col2:
            if i + 1 < len(activities):
                display_activity_card(activities[i + 1], i + 2)


def display_activity_card(activity: Dict[str, Any], index: int):
    """Display single activity card."""
    # Build a Google Maps search link for the activity name/address if available
    location_parts = []
    if isinstance(activity.get('location'), dict):
        addr = activity['location'].get('address')
        if addr:
            location_parts.append(addr)
    query = f"{activity.get('name','')} {' '.join(location_parts)}".strip().replace(' ', '+')
    maps_link = f"https://www.google.com/maps/search/{query}" if query else ""
    
    with st.container():
        st.markdown(f"""
        <div class="activity-card">
            <div style="display:flex; justify-content:space-between; align-items:flex-start; gap:0.5rem; flex-wrap:wrap;">
                <div>
                    <h4>{index}. {activity['name']}</h4>
                    <div class="mini-pill" style="margin-bottom:8px;">{activity['category'].title()}</div>
                    <p><strong>⭐ Rating:</strong> {activity.get('rating', 'N/A')}/5.0</p>
                    <p><strong>🕐 Best Time:</strong> {activity.get('best_time', 'Anytime')}</p>
                </div>
                <div class="pill" style="background:rgba(29,211,176,0.15); border-color:rgba(29,211,176,0.4); color:#a5f3fc;">
                    ${activity.get('price', 0) or 0:.2f}
                </div>
            </div>
            <p style="margin-top:0.5rem;"><strong>⏱️ Duration:</strong> {activity['duration_hours']} hours</p>
            <p><strong>What you'll do:</strong> {activity['description']}</p>
            {f'<p><a href="{maps_link}" target="_blank">🌐 View on Google Maps</a></p>' if maps_link else ''}
        </div>
        """, unsafe_allow_html=True)


def display_daily_schedule(schedule: List[Dict[str, Any]]):
    """Display day-by-day schedule."""
    
    # Use tabs for each day
    if schedule:
        tabs = st.tabs([f"Day {day['day']} - {day['date']}" for day in schedule])
        
        for i, day in enumerate(schedule):
            with tabs[i]:
                # Weather info
                st.markdown(f"**🌤️ Weather:** {day['weather']['condition']}")
                st.markdown(f"**🌡️ Temperature:** High {day['weather']['temperature']['high']}°C")
                
                st.markdown("---")
                
                # Events
                st.markdown("**📅 Schedule:**")
                for event in day['events']:
                    st.markdown(f"- **{event['time']}:** {event['activity']}")


def display_alternative_plans(alternatives: List[Dict[str, Any]]):
    """Display alternative budget plans."""
    
    st.markdown("### 💡 Alternative Plans")
    st.caption("Compare different budget allocations based on comfort level")
    
    tabs = st.tabs([alt['label'] for alt in alternatives])
    
    for i, alt in enumerate(alternatives):
        with tabs[i]:
            st.markdown(f"**{alt['description']}**")
            
            col1, col2, col3, col4 = st.columns(4)
            
            with col1:
                st.metric("Total Cost", f"${alt['total_cost']:,.2f}")
            with col2:
                st.metric("Balance", f"${alt['balance']:,.2f}")
            with col3:
                st.metric("Value Score", f"{alt['value_score']}/100")
            with col4:
                st.metric("Comfort", alt['comfort_level'].title())
            
            # Show what's included
            with st.expander("View Details"):
                selected = alt['selected_options']
                
                if selected['flight']:
                    st.markdown(f"**✈️ Flight:** {selected['flight'].get('airline', 'N/A')} - ${selected['flight']['price']:,.2f}")
                
                if selected['hotel']:
                    st.markdown(f"**🏨 Hotel:** {selected['hotel']['name']} - ${selected['hotel']['price']['total']:,.2f}")
                
                st.markdown(f"**🎯 Activities:** {len(selected['activities'])} activities")
