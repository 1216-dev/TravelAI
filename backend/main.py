"""
Main orchestration pipeline for the AI Travel Planner.
Coordinates all agents and manages the end-to-end workflow.
"""

import asyncio
import json
import time
import sys
import os
from typing import Dict, Any, Optional
from datetime import datetime, timedelta
from pathlib import Path

# Add project root to path
sys.path.insert(0, str(Path(__file__).parent.parent))

from backend.orchestrator import create_orchestrator
from backend.api_manager import APIManager
from backend.personalization_gnn import create_personalization_agent
from backend.budget_optimizer import create_budget_optimizer
from backend.itinerary_agent import create_itinerary_agent
from backend.utils.logger import get_logger, log_agent_activity
from backend.utils.validators import validate_request
from backend.utils.airport_codes import get_airport_code

logger = get_logger(__name__)


class TravelPlannerPipeline:
    """Main pipeline that orchestrates all AI agents."""
    
    def __init__(self):
        """Initialize the pipeline with all agents."""
        logger.info("🚀 Initializing AI Travel Planner Pipeline")
        
        self.orchestrator = create_orchestrator()
        self.api_manager = APIManager()
        self.personalization_agent = create_personalization_agent()
        self.budget_optimizer = create_budget_optimizer()
        self.itinerary_agent = create_itinerary_agent()
        
        logger.info("✅ All agents initialized successfully")
    
    async def process_request(
        self,
        request: Dict[str, Any],
        progress_callback: Optional[callable] = None
    ) -> Dict[str, Any]:
        """
        Process a complete travel planning request.
        
        Args:
            request: Travel planning request dictionary
            progress_callback: Optional callback function for progress updates
            
        Returns:
            Complete travel plan with itinerary
        """
        start_time = time.time()
        logger.info(f"📋 Processing travel request for {request.get('destination')}")
        
        result = {
            'success': False,
            'request_id': f"REQ-{datetime.now().strftime('%Y%m%d%H%M%S')}",
            'errors': [],
            'warnings': [],
            'data': {}
        }
        
        try:
            # Stage 1: Validation
            self._update_progress(progress_callback, "validation", "Validating request", 10)
            validation = self.orchestrator.validate_request(request)
            
            if not validation['is_valid']:
                result['errors'] = validation['errors']
                result['warnings'] = validation['warnings']
                logger.error(f"❌ Validation failed: {validation['errors']}")
                return result
            
            result['warnings'] = validation['warnings']
            
            # Stage 2: Intent Parsing
            self._update_progress(progress_callback, "intent_parsing", "Analyzing travel intent", 20)
            intent = self.orchestrator.parse_user_intent(request)
            logger.info(f"🎯 Intent parsed: {intent['trip_type']} trip")
            
            # Stage 3: Workflow Planning
            self._update_progress(progress_callback, "workflow_planning", "Planning agent workflow", 25)
            workflow = self.orchestrator.plan_workflow(request, intent)
            logger.info(f"📊 Workflow planned: {len(workflow['stages'])} stages")
            
            # Stage 4: Data Collection (35-55%)
            self._update_progress(progress_callback, "data_collection", "Collecting flight, hotel, and weather data", 35)
            
            # Extract dates from request (handle both formats)
            start_date = request.get('start_date') or request.get('dates', {}).get('start')
            end_date = request.get('end_date') or request.get('dates', {}).get('end')
            
            # Extract budget (handle both formats)
            budget_value = request.get('budget')
            if isinstance(budget_value, dict):
                budget_value = budget_value.get('total', 2500)
            elif budget_value is None:
                budget_value = 2500
            
            # Extract comfort level for flight class
            preferences_data = request.get('preferences', {})
            comfort_level = preferences_data.get('comfort_level', 'standard')
            
            # Extract min hotel rating
            min_hotel_rating = request.get('constraints', {}).get('hotel_rating_min', 3.5)
            
            # Convert city names to airport codes for flight API
            origin_code = get_airport_code(request['origin'])
            destination_code = get_airport_code(request['destination'])
            
            logger.info(f"✈️ Converting locations: {request['origin']} → {origin_code}, {request['destination']} → {destination_code}")
            
            flights, hotels, weather_forecasts = await asyncio.gather(
                self.api_manager.fetch_flights(
                    origin_code,  # Use airport code
                    destination_code,  # Use airport code
                    start_date,
                    end_date,
                    request.get('group_size', 1),
                    budget_value * 0.35,  # Allocate 35% of budget to flights
                    comfort_level  # Pass comfort level for cabin class
                ),
                self.api_manager.fetch_hotels(
                    request['destination'],
                    start_date,
                    end_date,
                    request.get('group_size', 1),
                    min_hotel_rating,
                    budget_value * 0.35  # Allocate 35% of budget to hotels
                ),
                self._fetch_weather_data(request)
            )
            
            # Fetch activities using the new API
            activities = await self.api_manager.fetch_activities(
                request['destination'],
                request.get('preferences', {}).get('categories', []),
                limit=20
            )
            
            logger.info(f"✅ Data collection complete: {len(flights)} flights, {len(hotels)} hotels, {len(activities)} activities")
            
            # Stage 5: Personalization (55-65%)
            self._update_progress(progress_callback, "personalization", "Personalizing recommendations", 55)
            personalized_activities = self.personalization_agent.recommend_activities(
                user_profile={
                    'preferences': request.get('preferences', []),
                    'budget': request.get('budget', 1000),
                    'group_size': request.get('group_size', 1)
                },
                destination=request['destination'],
                available_activities=activities  # Use real activities
            )
            
            # For flights and hotels, just use the fetched data directly
            # The budget optimizer will select the best options
            ranked_flights = flights
            ranked_hotels = hotels
            
            logger.info(f"✅ Personalization complete: {len(personalized_activities)} activities recommended")
            
            # Stage 6: Budget Optimization (70-85%)
            self._update_progress(progress_callback, "budget_optimization", "Optimizing budget allocation", 70)
            
            # Calculate trip duration (handle both date formats)
            start_date_str = request.get('start_date') or request.get('dates', {}).get('start')
            end_date_str = request.get('end_date') or request.get('dates', {}).get('end')
            start_dt = datetime.strptime(start_date_str, "%Y-%m-%d")
            end_dt = datetime.strptime(end_date_str, "%Y-%m-%d")
            duration = (end_dt - start_dt).days
            
            # Extract budget (handle both formats)
            budget_value = request.get('budget')
            if isinstance(budget_value, dict):
                budget_value = budget_value.get('total', 2500)
            elif budget_value is None:
                budget_value = 2500
            
            # Extract preferences
            preferences_data = request.get('preferences', {})
            
            optimized_budget = self.budget_optimizer.optimize_budget(
                budget_value,
                duration,
                ranked_flights,
                ranked_hotels,
                personalized_activities,
                preferences_data
            )
            
            alternatives = self.budget_optimizer.generate_alternatives(
                budget_value,
                duration,
                ranked_flights,
                ranked_hotels,
                personalized_activities,
                preferences_data
            )
            
            logger.info(f"✅ Budget optimized: ${optimized_budget['total_cost']:.2f} / ${optimized_budget['total_budget']:.2f}")
            
            # Stage 7: Itinerary Generation (85-95%)
            self._update_progress(progress_callback, "itinerary_generation", "Generating final itinerary", 85)
            
            # Convert weather dict to list format expected by itinerary agent
            weather_list = [{'date': date, **weather} for date, weather in weather_forecasts.items()]
            
            itinerary = self.itinerary_agent.consolidate_itinerary(
                request,
                optimized_budget,
                weather_list,
                request.get('user_id', 'user_unknown')
            )
            
            # Stage 8: PDF Generation (95-100%)
            self._update_progress(progress_callback, "pdf_generation", "Creating PDF document", 95)
            pdf_path = self.itinerary_agent.generate_pdf(itinerary)
            calendar_path = self.itinerary_agent.export_calendar_events(itinerary)
            
            logger.info(f"✅ Itinerary generated: {itinerary['itinerary_id']}")
            logger.info(f"📄 PDF: {pdf_path}")
            logger.info(f"📅 Calendar: {calendar_path}")
            
            # Stage 9: Complete
            self._update_progress(progress_callback, "complete", "Travel plan ready!", 100)
            
            result['success'] = True
            result['data'] = {
                'itinerary': itinerary,
                'alternatives': alternatives,
                'intent': intent,
                'pdf_path': pdf_path,
                'calendar_path': calendar_path,
                'processing_time': time.time() - start_time
            }
            
            logger.info(f"🎉 Request processed successfully in {result['data']['processing_time']:.2f}s")
            
        except Exception as e:
            logger.error(f"❌ Pipeline error: {str(e)}", exc_info=True)
            result['errors'].append(f"Pipeline error: {str(e)}")
        
        return result
        
        return result
    
    async def _fetch_weather_data(self, request: Dict[str, Any]) -> Dict[str, Any]:
        """Fetch weather data for the trip dates."""
        start_date_str = request.get('start_date') or request.get('dates', {}).get('start')
        end_date_str = request.get('end_date') or request.get('dates', {}).get('end')
        
        start_dt = datetime.strptime(start_date_str, "%Y-%m-%d")
        end_dt = datetime.strptime(end_date_str, "%Y-%m-%d")
        
        weather_forecasts = {}
        current_date = start_dt
        
        while current_date <= end_dt:
            date_str = current_date.strftime("%Y-%m-%d")
            weather = await self.api_manager.fetch_weather(
                request['destination'],
                date_str
            )
            weather_forecasts[date_str] = weather
            current_date += timedelta(days=1)
        
        return weather_forecasts
    
    def _update_progress(
        self,
        callback: Optional[callable],
        stage: str,
        message: str,
        progress: int
    ):
        """Update progress via callback."""
        if callback:
            callback({
                'stage': stage,
                'message': message,
                'progress': progress
            })
        logger.info(f"[{progress}%] {stage}: {message}")


async def process_travel_request(request: Dict[str, Any]) -> Dict[str, Any]:
    """
    Main entry point for processing travel requests.
    
    Args:
        request: Travel planning request
        
    Returns:
        Complete travel plan
    """
    pipeline = TravelPlannerPipeline()
    return await pipeline.process_request(request)


async def main():
    """Main function for testing the pipeline."""
    logger.info("=" * 80)
    logger.info("AI TRAVEL PLANNER - BACKEND TEST")
    logger.info("=" * 80)
    
    # Load sample request
    with open('backend/utils/sample_input.json', 'r') as f:
        sample_request = json.load(f)
    
    logger.info(f"\n📝 Sample Request:")
    logger.info(f"   Destination: {sample_request['destination']}")
    logger.info(f"   Dates: {sample_request['dates']['start']} to {sample_request['dates']['end']}")
    logger.info(f"   Budget: ${sample_request['budget']['total']}")
    logger.info(f"   Group Size: {sample_request['group_size']}")
    logger.info("")
    
    # Process request
    result = await process_travel_request(sample_request)
    
    # Display results
    if result['success']:
        logger.info("\n" + "=" * 80)
        logger.info("✅ SUCCESS - Travel Plan Generated")
        logger.info("=" * 80)
        
        itinerary = result['data']['itinerary']
        logger.info(f"\n📋 Itinerary ID: {itinerary['itinerary_id']}")
        logger.info(f"💰 Budget: ${itinerary['budget_summary']['total_cost']:.2f} / ${itinerary['budget_summary']['total_budget']:.2f}")
        logger.info(f"💵 Balance: ${itinerary['budget_summary']['balance']:.2f}")
        logger.info(f"⭐ Value Score: {itinerary['value_score']}")
        logger.info(f"📄 PDF: {result['data']['pdf_path']}")
        logger.info(f"📅 Calendar: {result['data']['calendar_path']}")
        logger.info(f"⏱️  Processing Time: {result['data']['processing_time']:.2f}s")
        
        # Save result to JSON
        output_file = f"output/result_{itinerary['itinerary_id']}.json"
        with open(output_file, 'w') as f:
            # Remove non-serializable objects
            clean_result = result.copy()
            json.dump(clean_result, f, indent=2, default=str)
        logger.info(f"💾 Full result saved to: {output_file}")
    else:
        logger.error("\n" + "=" * 80)
        logger.error("❌ FAILED - Travel Plan Generation")
        logger.error("=" * 80)
        logger.error(f"\nErrors: {result['errors']}")
        if result['warnings']:
            logger.warning(f"Warnings: {result['warnings']}")


if __name__ == "__main__":
    # Run the pipeline
    asyncio.run(main())
