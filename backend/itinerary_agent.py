"""
Itinerary Agent for consolidating travel plans and generating PDF documents.
Final stage that produces user-facing itinerary output.
"""

import os
import json
from datetime import datetime, timedelta
from typing import Dict, List, Any
from io import BytesIO
import yaml

from reportlab.lib.pagesizes import letter, A4
from reportlab.lib import colors
from reportlab.lib.units import inch
from reportlab.platypus import SimpleDocTemplate, Table, TableStyle, Paragraph, Spacer, PageBreak, Image
from reportlab.lib.styles import getSampleStyleSheet, ParagraphStyle
from reportlab.lib.enums import TA_CENTER, TA_LEFT, TA_RIGHT
from reportlab.pdfgen import canvas

from backend.utils.logger import get_logger

logger = get_logger(__name__)


class ItineraryAgent:
    """
    Consolidates travel recommendations and generates PDF itineraries.
    """
    
    def __init__(self, config_path: str = "backend/utils/config.yaml"):
        """Initialize itinerary agent."""
        with open(config_path, 'r') as f:
            self.config = yaml.safe_load(f)
        
        self.output_config = self.config['output']
        
        # Create output directory
        self.output_dir = "output/itineraries"
        os.makedirs(self.output_dir, exist_ok=True)
        
        logger.info("Itinerary Agent initialized")
    
    def consolidate_itinerary(
        self,
        request: Dict[str, Any],
        optimized_budget: Dict[str, Any],
        weather_data: List[Dict[str, Any]],
        user_id: str
    ) -> Dict[str, Any]:
        """
        Consolidate all travel components into a unified itinerary.
        
        Args:
            request: Original travel request
            optimized_budget: Budget optimization results
            weather_data: Weather forecasts
            user_id: User identifier
            
        Returns:
            Complete itinerary dictionary
        """
        logger.info("Consolidating travel itinerary")
        
        selected = optimized_budget['selected_options']
        flight_leg = self._format_flight_leg(selected['flight'], 'outbound') if selected['flight'] else None
        
        itinerary = {
            'itinerary_id': f"ITN-{datetime.now().strftime('%Y%m%d%H%M%S')}",
            'user_id': user_id,
            'generated_at': datetime.now().isoformat(),
            'trip_summary': {
                'destination': request['destination'],
                'origin': request.get('origin', 'N/A'),
                'dates': request['dates'],
                'duration_days': self._calculate_duration(request['dates']),
                'group_size': request.get('group_size', 1)
            },
            'budget_summary': {
                'total_budget': optimized_budget['total_budget'],
                'total_cost': optimized_budget['total_cost'],
                'balance': optimized_budget['balance'],
                'breakdown': optimized_budget['actual_costs']
            },
            'transportation': {
                'outbound_flight': flight_leg,
                'return_flight': flight_leg.get('return') if flight_leg else None
            },
            'accommodation': self._format_hotel(selected['hotel']) if selected['hotel'] else None,
            'activities': self._format_activities(selected['activities']),
            'weather_forecast': weather_data,
            'daily_schedule': self._generate_daily_schedule(
                request['dates'],
                selected['activities'],
                weather_data
            ),
            'recommendations': self._generate_recommendations(request, optimized_budget),
            'value_score': optimized_budget['value_score']
        }
        
        logger.info(f"Itinerary consolidated - ID: {itinerary['itinerary_id']}")
        return itinerary
    
    def generate_pdf(
        self,
        itinerary: Dict[str, Any],
        filename: str = None
    ) -> str:
        """
        Generate PDF document from itinerary.
        
        Args:
            itinerary: Complete itinerary dictionary
            filename: Optional custom filename
            
        Returns:
            Path to generated PDF file
        """
        logger.info("Generating PDF itinerary")
        
        if filename is None:
            filename = f"{itinerary['itinerary_id']}.pdf"
        
        filepath = os.path.join(self.output_dir, filename)
        
        # Create PDF document
        doc = SimpleDocTemplate(
            filepath,
            pagesize=letter,
            rightMargin=0.75*inch,
            leftMargin=0.75*inch,
            topMargin=0.75*inch,
            bottomMargin=0.75*inch
        )
        
        # Build PDF content
        story = []
        styles = getSampleStyleSheet()
        
        # Custom styles
        title_style = ParagraphStyle(
            'CustomTitle',
            parent=styles['Heading1'],
            fontSize=24,
            textColor=colors.HexColor('#1e3a8a'),
            spaceAfter=30,
            alignment=TA_CENTER
        )
        
        heading_style = ParagraphStyle(
            'CustomHeading',
            parent=styles['Heading2'],
            fontSize=16,
            textColor=colors.HexColor('#2563eb'),
            spaceAfter=12,
            spaceBefore=12
        )
        
        # Title
        story.append(Paragraph("✈️ Your Travel Itinerary", title_style))
        story.append(Spacer(1, 0.2*inch))
        
        # Trip Summary
        story.append(Paragraph("Trip Summary", heading_style))
        summary = itinerary['trip_summary']
        summary_data = [
            ['Destination:', summary['destination']],
            ['Dates:', f"{summary['dates']['start']} to {summary['dates']['end']}"],
            ['Duration:', f"{summary['duration_days']} days"],
            ['Travelers:', str(summary['group_size'])],
            ['Itinerary ID:', itinerary['itinerary_id']]
        ]
        summary_table = Table(summary_data, colWidths=[2*inch, 4*inch])
        summary_table.setStyle(TableStyle([
            ('FONTNAME', (0, 0), (-1, -1), 'Helvetica'),
            ('FONTSIZE', (0, 0), (-1, -1), 10),
            ('TEXTCOLOR', (0, 0), (0, -1), colors.HexColor('#1e3a8a')),
            ('ALIGN', (0, 0), (0, -1), 'RIGHT'),
            ('ALIGN', (1, 0), (1, -1), 'LEFT'),
            ('VALIGN', (0, 0), (-1, -1), 'MIDDLE'),
            ('BOTTOMPADDING', (0, 0), (-1, -1), 8),
        ]))
        story.append(summary_table)
        story.append(Spacer(1, 0.3*inch))
        
        # Budget Summary
        story.append(Paragraph("Budget Summary", heading_style))
        budget = itinerary['budget_summary']
        budget_data = [
            ['Category', 'Amount'],
            ['Total Budget', f"${budget['total_budget']:.2f}"],
            ['Total Cost', f"${budget['total_cost']:.2f}"],
            ['Balance', f"${budget['balance']:.2f}"]
        ]
        budget_table = Table(budget_data, colWidths=[3*inch, 3*inch])
        budget_table.setStyle(TableStyle([
            ('BACKGROUND', (0, 0), (-1, 0), colors.HexColor('#2563eb')),
            ('TEXTCOLOR', (0, 0), (-1, 0), colors.whitesmoke),
            ('ALIGN', (0, 0), (-1, -1), 'CENTER'),
            ('FONTNAME', (0, 0), (-1, 0), 'Helvetica-Bold'),
            ('FONTSIZE', (0, 0), (-1, 0), 12),
            ('BOTTOMPADDING', (0, 0), (-1, 0), 12),
            ('GRID', (0, 0), (-1, -1), 1, colors.grey),
            ('FONTNAME', (0, 1), (-1, -1), 'Helvetica'),
            ('FONTSIZE', (0, 1), (-1, -1), 10),
        ]))
        story.append(budget_table)
        story.append(Spacer(1, 0.3*inch))
        
        # Flights
        if itinerary['transportation']['outbound_flight']:
            story.append(Paragraph("✈️ Flights", heading_style))
            
            outbound = itinerary['transportation']['outbound_flight']
            story.append(Paragraph(f"<b>Outbound:</b> {outbound['departure']['airport']} → {outbound['arrival']['airport']}", styles['Normal']))
            story.append(Paragraph(f"Departure: {outbound['departure']['time']} | Arrival: {outbound['arrival']['time']}", styles['Normal']))
            story.append(Paragraph(f"Airline: {outbound.get('airline', 'N/A')} | Cabin: {outbound.get('cabin_class', 'Economy')}", styles['Normal']))
            story.append(Spacer(1, 0.1*inch))
            
            return_flight = itinerary['transportation']['return_flight']
            if return_flight:
                story.append(Paragraph(f"<b>Return:</b> {return_flight['departure']['airport']} → {return_flight['arrival']['airport']}", styles['Normal']))
                story.append(Paragraph(f"Departure: {return_flight['departure']['time']} | Arrival: {return_flight['arrival']['time']}", styles['Normal']))
            
            story.append(Spacer(1, 0.3*inch))
        
        # Accommodation
        if itinerary['accommodation']:
            story.append(Paragraph("🏨 Accommodation", heading_style))
            hotel = itinerary['accommodation']
            story.append(Paragraph(f"<b>{hotel['name']}</b> ({hotel['rating']}⭐)", styles['Normal']))
            story.append(Paragraph(f"Address: {hotel['location']['address']}", styles['Normal']))
            story.append(Paragraph(f"Room Type: {hotel['room_type']}", styles['Normal']))
            story.append(Paragraph(f"Price: ${hotel['price']['total']:.2f} ({hotel['price']['nights']} nights)", styles['Normal']))
            story.append(Spacer(1, 0.3*inch))
        
        # Activities
        if itinerary['activities']:
            story.append(Paragraph("🎯 Recommended Activities", heading_style))
            for i, activity in enumerate(itinerary['activities'][:10], 1):  # Limit to 10 activities
                price_val = activity.get('price', 0) or 0
                story.append(Paragraph(f"{i}. <b>{activity['name']}</b> - ${price_val:.2f}", styles['Normal']))
                story.append(Paragraph(f"   {activity['description']} ({activity['duration_hours']} hours)", styles['Normal']))
                story.append(Spacer(1, 0.1*inch))
            
            story.append(Spacer(1, 0.3*inch))
        
        # Daily Schedule
        story.append(PageBreak())
        story.append(Paragraph("📅 Daily Schedule", heading_style))
        
        for day in itinerary['daily_schedule']:
            story.append(Paragraph(f"<b>Day {day['day']}: {day['date']}</b>", styles['Heading3']))
            story.append(Paragraph(f"Weather: {day['weather']['condition']}, {day['weather']['temperature']['high']}°C", styles['Normal']))
            
            for event in day['events']:
                story.append(Paragraph(f"• {event['time']}: {event['activity']}", styles['Normal']))
            
            story.append(Spacer(1, 0.2*inch))
        
        # Build PDF
        doc.build(story)
        
        logger.info(f"PDF generated: {filepath}")
        return filepath
    
    def export_calendar_events(
        self,
        itinerary: Dict[str, Any],
        filename: str = None
    ) -> str:
        """
        Export itinerary as calendar events (.ics file).
        
        Args:
            itinerary: Complete itinerary dictionary
            filename: Optional custom filename
            
        Returns:
            Path to generated .ics file
        """
        logger.info("Generating calendar events")
        
        if filename is None:
            filename = f"{itinerary['itinerary_id']}.ics"
        
        filepath = os.path.join(self.output_dir, filename)
        
        # Simple ICS format (in production, use icalendar library)
        ics_content = ["BEGIN:VCALENDAR", "VERSION:2.0", "PRODID:-//AI Travel Planner//EN"]
        
        # Add flight events
        if itinerary['transportation']['outbound_flight']:
            outbound = itinerary['transportation']['outbound_flight']
            ics_content.extend([
                "BEGIN:VEVENT",
                f"SUMMARY:Flight to {itinerary['trip_summary']['destination']}",
                f"DTSTART:{outbound['departure']['time'].replace('-', '').replace(':', '')}",
                f"DTEND:{outbound['arrival']['time'].replace('-', '').replace(':', '')}",
                f"LOCATION:{outbound['departure']['airport']}",
                f"DESCRIPTION:Outbound flight - {outbound.get('airline', 'N/A')}",
                "END:VEVENT"
            ])
        
        ics_content.append("END:VCALENDAR")
        
        with open(filepath, 'w') as f:
            f.write('\n'.join(ics_content))
        
        logger.info(f"Calendar events exported: {filepath}")
        return filepath
    
    def _calculate_duration(self, dates: Dict[str, str]) -> int:
        """Calculate trip duration in days."""
        start = datetime.strptime(dates['start'], "%Y-%m-%d")
        end = datetime.strptime(dates['end'], "%Y-%m-%d")
        return (end - start).days
    
    def _format_flight_leg(self, flight: Dict[str, Any], leg: str) -> Dict[str, Any]:
        """
        Format flight leg information with all required fields for frontend display.
        Returns the complete flight object with outbound/return structure.
        """
        # Return the entire flight object with all details
        # The frontend expects: total_price, travel_class, carbon_emissions at top level
        # Plus outbound and return objects with full details
        return {
            'total_price': flight.get('total_price', flight.get('price', 0)),
            'travel_class': flight.get('travel_class', flight.get('cabin_class', 'Economy')),
            'carbon_emissions': flight.get('carbon_emissions', 0),
            'airline': flight.get('airline', 'N/A'),
            'airline_logo': flight.get('airline_logo', ''),
            'flight_number': flight.get('flight_id', 'N/A'),
            'price': flight.get('price', flight.get('total_price', 0)),
            'cabin_class': flight.get('cabin_class', flight.get('travel_class', 'Economy')),
            'booking_url': flight.get('booking_url', ''),
            
            # Complete outbound leg with all details
            'outbound': flight.get('outbound', {}),
            
            # Complete return leg with all details  
            'return': flight.get('return', {}),
            
            # Legacy fields for backward compatibility
            'departure': flight.get('outbound', {}).get('departure', {}),
            'arrival': flight.get('outbound', {}).get('arrival', {}),
            'duration_hours': flight.get('outbound', {}).get('duration_hours', 0),
            'stops': flight.get('outbound', {}).get('stops', 0),
            'layovers': flight.get('outbound', {}).get('layovers', [])
        }
    
    def _format_hotel(self, hotel: Dict[str, Any]) -> Dict[str, Any]:
        """Format hotel information."""
        return {
            'name': hotel.get('name'),
            'rating': hotel.get('rating'),
            'type': hotel.get('type'),
            'location': hotel.get('location', {}),
            'room_type': hotel.get('room_type'),
            'price': {
                'per_night': hotel.get('price', {}).get('per_night'),
                'total': hotel.get('price', {}).get('total'),
                'nights': hotel.get('price', {}).get('nights', 0),
                'currency': hotel.get('price', {}).get('currency', 'USD')
            },
            'amenities': hotel.get('amenities', []),
            'policies': hotel.get('policies', {})
        }
    
    def _format_activities(self, activities: List[Dict[str, Any]]) -> List[Dict[str, Any]]:
        """Format activity information."""
        return [{
            'name': activity.get('name'),
            'category': activity.get('category'),
            'description': activity.get('description'),
            'duration_hours': activity.get('duration_hours'),
            'price': activity.get('price'),
            'rating': activity.get('rating'),
            'best_time': activity.get('best_time')
        } for activity in activities]
    
    def _generate_daily_schedule(
        self,
        dates: Dict[str, str],
        activities: List[Dict[str, Any]],
        weather_data: List[Dict[str, Any]]
    ) -> List[Dict[str, Any]]:
        """Generate day-by-day schedule."""
        start = datetime.strptime(dates['start'], "%Y-%m-%d")
        end = datetime.strptime(dates['end'], "%Y-%m-%d")
        duration = (end - start).days
        
        schedule = []
        activities_per_day = len(activities) // duration if duration > 0 else 0
        
        for day in range(duration):
            current_date = start + timedelta(days=day)
            date_str = current_date.strftime("%Y-%m-%d")
            
            # Get weather for this day
            weather = next((w for w in weather_data if w['date'] == date_str), {
                'condition': 'Unknown',
                'temperature': {'high': 'N/A', 'low': 'N/A'}
            })
            
            # Assign activities for this day
            day_start = day * activities_per_day
            day_end = day_start + activities_per_day
            day_activities = activities[day_start:day_end] if activities else []
            
            # Generate events
            events = [
                {'time': '08:00', 'activity': 'Breakfast'},
                {'time': '09:00', 'activity': 'Check-in / Start exploring'}
            ]
            
            # Add scheduled activities
            time_slots = ['10:00', '13:00', '16:00', '19:00']
            for i, activity in enumerate(day_activities[:len(time_slots)]):
                events.append({
                    'time': time_slots[i],
                    'activity': activity['name']
                })
            
            events.append({'time': '21:00', 'activity': 'Dinner & Leisure'})
            
            schedule.append({
                'day': day + 1,
                'date': date_str,
                'day_of_week': current_date.strftime("%A"),
                'weather': weather,
                'events': events
            })
        
        return schedule
    
    def _generate_recommendations(
        self,
        request: Dict[str, Any],
        optimized_budget: Dict[str, Any]
    ) -> List[str]:
        """Generate travel recommendations and tips."""
        recommendations = [
            "💡 Book flights and hotels as soon as possible for best rates",
            "📱 Download offline maps of your destination",
            "💳 Notify your bank about international travel",
            "🩹 Pack a basic first-aid kit",
            "📸 Don't forget travel adapters for your electronics"
        ]
        
        # Budget-specific recommendations
        if optimized_budget['balance'] > 200:
            recommendations.append("💰 You have budget surplus - consider upgrading accommodation or adding activities")
        elif optimized_budget['balance'] < 0:
            recommendations.append("⚠️ Budget is tight - look for free activities or adjust plans")
        
        return recommendations


def create_itinerary_agent() -> ItineraryAgent:
    """Factory function to create itinerary agent."""
    return ItineraryAgent()
