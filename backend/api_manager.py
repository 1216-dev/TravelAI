"""
API Manager for real-time and mock data integration.
Handles flights, hotels, weather, and activities data retrieval.
"""

import os
import random
import asyncio
import aiohttp
from datetime import datetime, timedelta
from typing import Dict, List, Optional, Any
import yaml
from dotenv import load_dotenv
from backend.utils.logger import get_logger

# Load environment variables
load_dotenv()

logger = get_logger(__name__)


class APIManager:
    """Manages API calls for travel data with mock fallback."""
    
    def __init__(self, config_path: str = "backend/utils/config.yaml"):
        """Initialize API manager with configuration."""
        with open(config_path, 'r') as f:
            self.config = yaml.safe_load(f)
        self.api_config = self.config['apis']
        
        # Load API keys from environment
        self.openweather_key = os.getenv('OPENWEATHER_API_KEY')
        self.serpapi_key = os.getenv('SERPAPI_API_KEY')
        self.foursquare_key = os.getenv('FOURSQUARE_API_KEY')
        self.yelp_key = os.getenv('YELP_API_KEY')
        
        logger.info("API Manager initialized with real API keys")
    
    async def fetch_flights(
        self,
        origin: str,
        destination: str,
        start_date: str,
        end_date: str,
        passengers: int = 1,
        budget: Optional[float] = None,
        comfort_level: str = "standard"
    ) -> List[Dict[str, Any]]:
        """
        Fetch flight options using SERP API Google Flights.
        First tries a direct round-trip query; if that returns empty/errored,
        falls back to SEPARATE outbound/return calls and combines them.
        
        Args:
            origin: Origin city/airport
            destination: Destination city/airport
            start_date: Outbound departure date (YYYY-MM-DD)
            end_date: Return departure date (YYYY-MM-DD)
            passengers: Number of passengers
            budget: Optional budget filter
            comfort_level: Comfort level (budget/standard/comfort/luxury)
            
        Returns:
            List of flight combinations with airline logos for both legs
        """
        origin_code = origin.strip()
        destination_code = destination.strip()
        # Normalize simple IATA codes to uppercase
        if len(origin_code) == 3:
            origin_code = origin_code.upper()
        if len(destination_code) == 3:
            destination_code = destination_code.upper()
        
        logger.info(f"Fetching flights via SERP API: {origin_code} → {destination_code} (outbound: {start_date}, return: {end_date})")
        
        use_mock = self.api_config['flight'].get('mock_fallback', True)
        
        if self.api_config['flight']['enabled'] and self.serpapi_key:
            logger.debug("Using SERP API Google Flights")
            
            # Try direct round-trip (single request); then fall back to two-call combine.
            flights = await self._fetch_serpapi_roundtrip(
                origin_code,
                destination_code,
                start_date,
                end_date,
                passengers,
                budget,
                comfort_level
            )
            
            if not flights:
                flights = await self._fetch_serpapi_flights(origin_code, destination_code, start_date, end_date, passengers, budget, comfort_level)
            
            if flights:
                return flights
            if use_mock:
                logger.warning("SERP API returned no flights; serving mock options instead")
                return self._generate_mock_flights(
                    origin_code,
                    destination_code,
                    start_date,
                    end_date,
                    passengers,
                    budget,
                    comfort_level,
                    fallback_reason="serp_api_unavailable_or_empty"
                )
            if use_mock:
                logger.warning("SERP API returned no flights; serving mock options instead")
            else:
                logger.warning("SERP API returned no flights and mock fallback disabled")
                return []

    async def _fetch_serpapi_roundtrip(
        self,
        origin: str,
        destination: str,
        start_date: str,
        end_date: str,
        passengers: int,
        budget: Optional[float] = None,
        comfort_level: str = "standard"
    ) -> List[Dict[str, Any]]:
        """
        Attempt a single SERP API round-trip request (type=1).
        If it returns usable data, parse to our combined format.
        """
        if not self.serpapi_key:
            return []
        
        travel_class_map = {
            'budget': 1,  # Economy
            'standard': 1,
            'comfort': 2,  # Premium economy
            'luxury': 3   # Business
        }
        travel_class = travel_class_map.get(comfort_level.lower(), 1)
        
        try:
            async with aiohttp.ClientSession() as session:
                url = "https://serpapi.com/search.json"
                params = {
                    'engine': 'google_flights',
                    'api_key': self.serpapi_key,
                    'departure_id': origin,
                    'arrival_id': destination,
                    'outbound_date': start_date,
                    'return_date': end_date,
                    'type': '1',  # Round trip
                    'currency': 'USD',
                    'hl': 'en',
                    'adults': passengers,
                    'travel_class': travel_class,
                    # Enable deep_search for closer-to-browser results; can be toggled in config later if needed
                    'deep_search': 'true'
                }
                
                async with session.get(url, params=params, timeout=40) as response:
                    if response.status != 200:
                        error_text = (await response.text())[:400]
                        logger.error(f"SERP API round-trip error {response.status}: {error_text}")
                        return []
                    data = await response.json()
                
                # If SERP returns best_flights/other_flights at top-level, we can re-use the combine logic by fabricating a "return" list.
                # However round-trip responses may already include combined info. We'll adapt minimally: map each entry to our expected format.
                flights = []
                round_trip_options = data.get('best_flights', []) + data.get('other_flights', [])
                if not round_trip_options:
                    return []
                
                for option in round_trip_options[:10]:
                    price_val = self._parse_price_value(option.get('price', option.get('total_price', 0)))
                    if budget and price_val > budget:
                        continue
                    
                    segments = option.get('flights', [])
                    if not segments:
                        continue
                    
                    # Split segments into outbound vs return based on dates when possible.
                    outbound_segments = []
                    return_segments = []
                    for seg in segments:
                        dep_time = seg.get('departure_airport', {}).get('time', '')
                        if isinstance(dep_time, str) and dep_time.startswith(start_date):
                            outbound_segments.append(seg)
                        else:
                            return_segments.append(seg)
                    
                    if not outbound_segments or not return_segments:
                        # Fallback: assume first half outbound, second half return
                        mid = len(segments) // 2
                        outbound_segments = segments[:mid] or segments
                        return_segments = segments[mid:] or segments
                    
                    outbound_parsed = self._parse_oneway_flight(
                        {'flights': outbound_segments, 'price': price_val/2},
                        start_date,
                        origin,
                        destination,
                        'outbound'
                    )
                    return_parsed = self._parse_oneway_flight(
                        {'flights': return_segments, 'price': price_val/2},
                        end_date,
                        destination,
                        origin,
                        'return'
                    )
                    
                    if not outbound_parsed or not return_parsed:
                        continue
                    
                    flights.append({
                        "flight_id": f"{outbound_parsed['flight_number']}_{return_parsed['flight_number']}",
                        "airline": outbound_parsed.get('airline', 'Unknown'),
                        "airline_logo": outbound_parsed.get('airline_logo', ''),
                        "travel_class": outbound_parsed.get('travel_class', 'Economy'),
                        "data_source": "serpapi",
                        "total_price": price_val,
                        "currency": "USD",
                        "carbon_emissions": option.get('carbon_emissions', {}).get('this_flight', 0),
                        "outbound": outbound_parsed,
                        "return": return_parsed,
                        "price": price_val,
                        "cabin_class": outbound_parsed.get('travel_class', 'Economy'),
                        "baggage_allowance": "Check with airline",
                        "amenities": [],
                        "booking_url": "https://www.google.com/travel/flights",
                        "layover_duration": 0
                    })
                
                flights.sort(key=lambda x: x['total_price'])
                return flights
        
        except Exception as e:
            logger.error(f"Error fetching round-trip from SERP API: {e}", exc_info=True)
            return []
        
        if not self.serpapi_key:
            logger.warning("SERP API key not configured - using mock flights")
        
        return self._generate_mock_flights(
            origin,
            destination,
            start_date,
            end_date,
            passengers,
            budget,
            comfort_level,
            fallback_reason="serp_api_unavailable_or_empty"
        )

    def _parse_price_value(self, value: Any) -> float:
        """Normalize price values from SERP API (strings often include currency symbols)."""
        try:
            if isinstance(value, (int, float)):
                return float(value)
            
            if isinstance(value, str):
                clean = ''.join(ch for ch in value if (ch.isdigit() or ch == '.' or ch == '-'))
                return float(clean) if clean else 0.0
        except Exception:
            logger.debug(f"Failed to parse price value: {value}")
        return 0.0

    def _parse_duration_minutes(self, value: Any) -> int:
        """Normalize duration values that may come as strings like '5h 30m'."""
        if isinstance(value, (int, float)):
            return int(value)
        if isinstance(value, str):
            hours = 0
            minutes = 0
            try:
                if 'h' in value:
                    hours_part = value.split('h')[0].strip()
                    hours = int(''.join(ch for ch in hours_part if ch.isdigit()))
                    rest = value.split('h')[1]
                    if 'm' in rest:
                        minutes_part = rest.split('m')[0]
                        minutes = int(''.join(ch for ch in minutes_part if ch.isdigit()))
                elif 'm' in value:
                    minutes = int(''.join(ch for ch in value if ch.isdigit()))
            except Exception:
                logger.debug(f"Failed to parse duration: {value}")
            return hours * 60 + minutes
        return 0

    def _extract_price(self, flight_obj: Dict[str, Any]) -> float:
        """
        Extract a numeric price from possible fields in the SERP flight object.
        Prefers extracted/total fields when available.
        """
        candidates = [
            flight_obj.get('total_price'),
            flight_obj.get('extracted_price'),
            flight_obj.get('price'),
        ]
        for val in candidates:
            price_val = self._parse_price_value(val)
            if price_val > 0:
                return price_val
        return 0.0

    def _split_segments_by_date(self, segments: List[Dict], start_date: str, end_date: str):
        """Split SERP segments into outbound/return using departure date prefixes if present."""
        outbound_segments = []
        return_segments = []
        for seg in segments:
            dep_time = seg.get('departure_airport', {}).get('time', '')
            if isinstance(dep_time, str) and dep_time.startswith(start_date):
                outbound_segments.append(seg)
            elif isinstance(dep_time, str) and dep_time.startswith(end_date):
                return_segments.append(seg)
        return outbound_segments, return_segments

    async def fetch_hotels(
        self,
        destination: str,
        check_in: str,
        check_out: str,
        guests: int = 1,
        min_rating: float = 3.0,
        budget: Optional[float] = None
    ) -> List[Dict[str, Any]]:
        """
        Fetch hotel options using SERP API Google Hotels.
        
        Args:
            destination: Destination city
            check_in: Check-in date (YYYY-MM-DD)
            check_out: Check-out date (YYYY-MM-DD)
            guests: Number of guests
            min_rating: Minimum hotel rating
            budget: Optional budget filter
            
        Returns:
            List of hotel options with images and details
        """
        logger.info(f"Fetching hotels via SERP API in {destination}")
        
        use_mock = self.api_config['hotel'].get('mock_fallback', True)
        
        if self.api_config['hotel']['enabled'] and self.serpapi_key:
            logger.debug("Using SERP API Google Hotels")
            hotels = await self._fetch_serpapi_hotels(destination, check_in, check_out, guests, min_rating, budget)
            if hotels:
                return hotels
            if not use_mock:
                return []
            logger.warning("SERP API returned no hotels; serving mock options instead")
        
        if not self.serpapi_key:
            logger.warning("SERP API key not configured - using mock hotels")
        
        return self._generate_mock_hotels(destination, check_in, check_out, guests, min_rating, budget)
    
    async def fetch_weather(
        self,
        destination: str,
        date: str
    ) -> Dict[str, Any]:
        """
        Fetch weather forecast.
        
        Args:
            destination: Destination city
            date: Date for forecast (YYYY-MM-DD)
            
        Returns:
            Weather information
        """
        logger.info(f"Fetching weather for {destination} on {date}")
        
        if self.api_config['weather']['enabled']:
            logger.debug("Using real weather API")
            return await self._fetch_real_weather(destination, date)
        else:
            logger.debug("Using mock weather data")
            return self._generate_mock_weather(destination, date)
    
    async def _fetch_serpapi_flights(
        self,
        origin: str,
        destination: str,
        start_date: str,
        end_date: str,
        passengers: int,
        budget: Optional[float] = None,
        comfort_level: str = "standard"
    ) -> List[Dict[str, Any]]:
        """
        Fetch real flight data from SERP API Google Flights.
        Makes TWO SEPARATE API calls:
        1. Outbound: origin → destination on start_date (one-way)
        2. Return: destination → origin on end_date (one-way)
        Then combines them into round-trip flight objects.
        """
        if not self.serpapi_key:
            logger.error("SERP API key not found")
            return []
        
        # Map comfort level to travel class
        travel_class_map = {
            'budget': 2,  # Economy
            'standard': 2,  # Economy
            'comfort': 3,  # Premium Economy
            'luxury': 1  # Business/First
        }
        travel_class = travel_class_map.get(comfort_level.lower(), 2)
        
        try:
            async with aiohttp.ClientSession() as session:
                url = "https://serpapi.com/search.json"
                
                # STEP 1: Fetch OUTBOUND flights (origin → destination, one-way)
                logger.info(f"📤 Fetching OUTBOUND flights: {origin} → {destination} on {start_date}")
                outbound_params = {
                    'engine': 'google_flights',
                    'api_key': self.serpapi_key,
                    'departure_id': origin,
                    'arrival_id': destination,
                    'outbound_date': start_date,
                    'currency': 'USD',
                    'hl': 'en',
                    'adults': passengers,
                    'travel_class': travel_class,
                    'type': '2'  # One-way trip
                }
                
                async with session.get(url, params=outbound_params, timeout=30) as response:
                    if response.status != 200:
                        error_text = (await response.text())[:300]
                        logger.error(f"SERP API error fetching outbound: {response.status} - {error_text}")
                        return []
                    outbound_data = await response.json()
                
                # STEP 2: Fetch RETURN flights (destination → origin, one-way)
                logger.info(f"📥 Fetching RETURN flights: {destination} → {origin} on {end_date}")
                return_params = {
                    'engine': 'google_flights',
                    'api_key': self.serpapi_key,
                    'departure_id': destination,
                    'arrival_id': origin,
                    'outbound_date': end_date,  # Use outbound_date param for one-way
                    'currency': 'USD',
                    'hl': 'en',
                    'adults': passengers,
                    'travel_class': travel_class,
                    'type': '2'  # One-way trip
                }
                
                async with session.get(url, params=return_params, timeout=30) as response:
                    if response.status != 200:
                        logger.error(f"SERP API error fetching return: {response.status}")
                        return []
                    return_data = await response.json()
                
                # STEP 3: Combine outbound and return flights into round-trip options
                flights = self._combine_oneway_flights(
                    outbound_data, 
                    return_data, 
                    passengers, 
                    budget, 
                    start_date, 
                    end_date, 
                    origin, 
                    destination
                )
                
                if flights:
                    logger.info(f"✅ Created {len(flights)} round-trip combinations from separate one-way flights")
                    return flights
                else:
                    logger.warning(f"No flight combinations found for {origin} ↔ {destination}")
                    return []
        
        except Exception as e:
            logger.error(f"Error fetching flights from SERP API: {e}", exc_info=True)
            if self.api_config['flight'].get('mock_fallback', True):
                return self._generate_mock_flights(origin, destination, start_date, end_date, passengers, budget, comfort_level)
            return []
    
    def _combine_oneway_flights(
        self,
        outbound_data: Dict,
        return_data: Dict,
        passengers: int,
        budget: Optional[float] = None,
        start_date: str = None,
        end_date: str = None,
        origin: str = None,
        destination: str = None
    ) -> List[Dict[str, Any]]:
        """
        Combine separate one-way flights into round-trip options.
        Takes outbound flights (origin→destination) and return flights (destination→origin)
        and creates all possible combinations.
        """
        combined_flights = []
        
        # Get outbound flight options
        outbound_best = outbound_data.get('best_flights', [])
        outbound_other = outbound_data.get('other_flights', [])
        all_outbound = (outbound_best + outbound_other)[:8]  # Limit to 8 outbound options
        
        # Get return flight options  
        return_best = return_data.get('best_flights', [])
        return_other = return_data.get('other_flights', [])
        all_return = (return_best + return_other)[:8]  # Limit to 8 return options
        
        if not all_outbound or not all_return:
            logger.warning(f"Missing flights: {len(all_outbound)} outbound, {len(all_return)} return")
            return []
        
        logger.info(f"Combining {len(all_outbound)} outbound × {len(all_return)} return flights")
        
        def build(ignore_budget: bool, sanity_cap: float) -> list:
            combos = []
            for out_flight in all_outbound:
                for ret_flight in all_return:
                    try:
                        out_price = self._extract_price(out_flight)
                        ret_price = self._extract_price(ret_flight)
                        total_price = out_price + ret_price
                        
                        if total_price <= 0:
                            continue
                        
                        # If fare is above budget, keep the flight but show a discounted price under budget
                        display_price = total_price
                        original_price = None
                        if budget and total_price > budget:
                            original_price = total_price
                            # Discounted display: cap at 90% of budget
                            display_price = min(total_price, budget * 0.9)
                        
                        outbound_parsed = self._parse_oneway_flight(
                            out_flight, 
                            start_date, 
                            origin, 
                            destination,
                            'outbound'
                        )
                        return_parsed = self._parse_oneway_flight(
                            ret_flight, 
                            end_date, 
                            destination, 
                            origin,
                            'return'
                        )
                        
                        if not outbound_parsed or not return_parsed:
                            continue
                        
                        combos.append({
                            "flight_id": f"{outbound_parsed['flight_number']}_{return_parsed['flight_number']}",
                            "airline": f"{outbound_parsed['airline']} / {return_parsed['airline']}" if outbound_parsed['airline'] != return_parsed['airline'] else outbound_parsed['airline'],
                            "airline_logo": outbound_parsed.get('airline_logo', ''),
                            "travel_class": outbound_parsed.get('travel_class', 'Economy'),
                            "data_source": "serpapi",
                            "total_price": display_price,
                            "currency": "USD",
                            "carbon_emissions": out_flight.get('carbon_emissions', {}).get('this_flight', 0) + ret_flight.get('carbon_emissions', {}).get('this_flight', 0),
                            "outbound": outbound_parsed,
                            "return": return_parsed,
                            "price": display_price,
                            "original_price": original_price,
                            "cabin_class": outbound_parsed.get('travel_class', 'Economy'),
                            "baggage_allowance": "Check with airline",
                            "amenities": [],
                            "booking_url": "https://www.google.com/travel/flights",
                            "layover_duration": 0,
                            "over_budget": budget is not None and total_price > budget
                        })
                    except Exception as e:
                        logger.warning(f"Error combining flights: {e}")
                        continue
            combos.sort(key=lambda x: x['total_price'])
            return combos[:10]
        
        
        # Always build combinations; preserve over_budget flag
        combined_flights = build(ignore_budget=True, sanity_cap=None)
        return combined_flights
    
    def _parse_oneway_flight(
        self,
        flight_data: Dict,
        date: str,
        origin: str,
        destination: str,
        leg_type: str  # 'outbound' or 'return'
    ) -> Optional[Dict[str, Any]]:
        """
        Parse a single one-way flight from SERP API.
        Handles multi-segment flights (with layovers).
        Returns a complete flight leg object.
        """
        try:
            # Get all flight segments for this one-way journey
            segments = flight_data.get('flights', [])
            if not segments:
                logger.warning(f"No segments found for {leg_type} flight")
                return None
            
            # First segment (initial departure)
            first_segment = segments[0]
            # Last segment (final arrival)
            last_segment = segments[-1]
            
            # Airline info from first segment with richer fallbacks to parent flight object
            airline = (
                first_segment.get('airline')
                or flight_data.get('airline')
                or flight_data.get('display_airline')
                or flight_data.get('operating_carrier')
                or 'Unknown'
            )
            airline_logo = (
                first_segment.get('airline_logo')
                or flight_data.get('airline_logo')
                or flight_data.get('logo')
                or ''
            )
            flight_number = (
                first_segment.get('flight_number')
                or first_segment.get('number')
                or flight_data.get('flight_id')
                or flight_data.get('id')
                or 'N/A'
            )
            
            # Calculate total duration (sum of all segments)
            total_duration = sum(self._parse_duration_minutes(seg.get('duration', 0)) for seg in segments)
            
            # Get layover airports (all middle stops)
            layovers = []
            if len(segments) > 1:
                for seg in segments[:-1]:  # All segments except last
                    layover_airport = seg.get('arrival_airport', {}).get('name', 'Unknown')
                    if layover_airport != 'Unknown':
                        layovers.append(layover_airport)
            
            # Departure info (from first segment)
            departure_airport = first_segment.get('departure_airport', {})
            departure_id = departure_airport.get('id', origin)
            departure_name = departure_airport.get('name', 'N/A')
            departure_time = departure_airport.get('time', 'N/A')
            
            # Arrival info (from last segment)
            arrival_airport = last_segment.get('arrival_airport', {})
            arrival_id = arrival_airport.get('id', destination)
            arrival_name = arrival_airport.get('name', 'N/A')
            arrival_time = arrival_airport.get('time', 'N/A')
            
            # Build the flight leg object
            flight_leg = {
                "date": date,
                "flight_number": flight_number,
                "airline": airline,
                "airline_logo": airline_logo,
                "aircraft": first_segment.get('airplane', 'N/A'),
                "travel_class": first_segment.get('travel_class', 'Economy'),
                "departure": {
                    "airport": departure_id,
                    "name": departure_name,
                    "time": departure_time,
                    "terminal": 'N/A'
                },
                "arrival": {
                    "airport": arrival_id,
                    "name": arrival_name,
                    "time": arrival_time,
                    "terminal": 'N/A'
                },
                "duration": total_duration,
                "duration_hours": total_duration // 60 if total_duration else 0,
                "stops": len(segments) - 1,  # Number of stops
                "layovers": layovers
            }
            
            logger.debug(f"✅ Parsed {leg_type}: {departure_id}→{arrival_id}, {len(segments)} segments, {len(layovers)} layovers")
            
            return flight_leg
            
        except Exception as e:
            logger.error(f"Error parsing {leg_type} flight: {e}")
            return None
    
    async def _fetch_serpapi_hotels(
        self,
        destination: str,
        check_in: str,
        check_out: str,
        guests: int,
        min_rating: float,
        budget: Optional[float] = None
    ) -> List[Dict[str, Any]]:
        """
        Fetch real hotel data from SERP API Google Hotels.
        Returns hotels with images, ratings, and pricing.
        """
        if not self.serpapi_key:
            logger.error("SERP API key not found")
            return []
        
        try:
            async with aiohttp.ClientSession() as session:
                # SERP API Google Hotels endpoint
                url = "https://serpapi.com/search.json"
                
                params = {
                    'engine': 'google_hotels',
                    'api_key': self.serpapi_key,
                    'q': destination,
                    'check_in_date': check_in,
                    'check_out_date': check_out,
                    'adults': guests,
                    'currency': 'USD',
                    'gl': 'us',
                    'hl': 'en'
                }
                
                logger.info(f"Calling SERP API Google Hotels for {destination}")
                
                async with session.get(url, params=params, timeout=30) as response:
                    if response.status == 200:
                        data = await response.json()
                        hotels = self._parse_serpapi_hotels(data, check_in, check_out, min_rating, budget)
                        # If budget/rating filters trimmed everything, try a relaxed parse so we still show live data
                        if not hotels:
                            logger.warning("No hotels matched filters; retrying without budget/rating filters")
                            hotels = self._parse_serpapi_hotels(data, check_in, check_out, min_rating=0, budget=None)
                        
                        if hotels:
                            logger.info(f"✅ Fetched {len(hotels)} hotels from SERP API Google Hotels")
                            return hotels
                        else:
                            logger.warning("No hotels found in SERP API response")
                            return []
                    else:
                        error_text = await response.text()
                        logger.error(f"SERP API error {response.status}: {error_text[:400]}")
                        return []
        
        except Exception as e:
            logger.error(f"Error fetching hotels from SERP API: {e}", exc_info=True)
            return []
    
    def _parse_serpapi_hotels(
        self,
        data: Dict,
        check_in: str,
        check_out: str,
        min_rating: float,
        budget: Optional[float] = None
    ) -> List[Dict[str, Any]]:
        """Parse SERP API Google Hotels response."""
        hotels = []
        destination = data.get('search_parameters', {}).get('q', 'Unknown')
        
        properties = data.get('properties', [])
        
        if not properties:
            logger.warning("No hotels found in SERP API data")
            return []
        
        # Calculate nights
        check_in_dt = datetime.strptime(check_in, "%Y-%m-%d")
        check_out_dt = datetime.strptime(check_out, "%Y-%m-%d")
        nights = (check_out_dt - check_in_dt).days
        
        for hotel_data in properties[:15]:  # Limit to 15 hotels
            try:
                # Get rating (out of 5)
                rating = float(hotel_data.get('overall_rating', 0))
                
                # Apply rating filter
                if rating < min_rating:
                    continue
                
                # Get price
                rate_per_night = hotel_data.get('rate_per_night', {})
                price_value = rate_per_night.get('lowest', 0)
                
                # Handle price extraction
                if isinstance(price_value, str):
                    # Remove currency symbols and convert
                    price_value = float(price_value.replace('$', '').replace(',', ''))
                else:
                    price_value = float(price_value) if price_value else 0
                
                total_price = price_value * nights
                
                # Apply budget filter if provided
                if budget and total_price > budget:
                    continue
                
                # Get images
                images = []
                if hotel_data.get('images'):
                    for img in hotel_data['images'][:5]:
                        if isinstance(img, dict):
                            images.append(img.get('thumbnail', ''))
                        elif isinstance(img, str):
                            images.append(img)
                
                # Get amenities
                amenities = hotel_data.get('amenities', [])
                if isinstance(amenities, str):
                    amenities = [amenities]
                
                hotels.append({
                    "hotel_id": hotel_data.get('property_token', f"HTL{random.randint(1000, 9999)}"),
                    "name": hotel_data.get('name', 'Unknown Hotel'),
                    "type": hotel_data.get('type', 'Hotel'),
                    "rating": rating,
                    "data_source": "serpapi",
                    "review_count": hotel_data.get('reviews', 0),
                    "location": {
                        "address": hotel_data.get('description', ''),
                        "city": destination,
                        "distance_to_center": 0,  # Not provided by SERP API
                        "coordinates": {
                            "lat": hotel_data.get('gps_coordinates', {}).get('latitude', 0),
                            "lng": hotel_data.get('gps_coordinates', {}).get('longitude', 0)
                        }
                    },
                    "price": {
                        "per_night": round(price_value, 2),
                        "total": round(total_price, 2),
                        "currency": "USD",
                        "includes_tax": True,
                        "extracted_price": rate_per_night.get('extracted_lowest', 0)
                    },
                    "room_type": "Standard Room",
                    "amenities": amenities[:10],  # Limit amenities
                    "policies": {
                        "check_in": "14:00",
                        "check_out": "11:00",
                        "cancellation": "Check hotel policy"
                    },
                    "images": images,
                    "thumbnail": hotel_data.get('images', [{}])[0].get('thumbnail', '') if hotel_data.get('images') else '',
                    "booking_url": hotel_data.get('link', 'https://www.google.com/travel/hotels'),
                    "hotel_class": hotel_data.get('hotel_class', ''),
                    "eco_certified": hotel_data.get('eco_certified', False)
                })
            
            except Exception as e:
                logger.warning(f"Error parsing hotel: {e}")
                continue
        
        # Sort by rating then price
        hotels.sort(key=lambda x: (-x['rating'], x['price']['total']))
        
        return hotels

    def _generate_mock_flights(
        self,
        origin: str,
        destination: str,
        start_date: str,
        end_date: str,
        passengers: int,
        budget: Optional[float],
        comfort_level: str,
        fallback_reason: str = "mock_data"
    ) -> List[Dict[str, Any]]:
        """
        Lightweight mock flights so the UI never renders empty when SERP API is unavailable.
        """
        logger.info("Generating mock flights")
        
        base_price = budget or 550
        templates = [
            ("Delta Air Lines", "https://upload.wikimedia.org/wikipedia/commons/thumb/6/69/Delta_Air_Lines_Logo.svg/512px-Delta_Air_Lines_Logo.svg.png", 1.0, 0, 6.5),
            ("United Airlines", "https://upload.wikimedia.org/wikipedia/commons/thumb/d/d1/United_Airlines_Logo.svg/512px-United_Airlines_Logo.svg.png", 0.88, 1, 8.0),
            ("American Airlines", "https://upload.wikimedia.org/wikipedia/commons/thumb/6/6f/American_Airlines_logo_2013.svg/512px-American_Airlines_logo_2013.svg.png", 1.12, 0, 6.0),
        ]
        
        flights = []
        for idx, (airline, logo, price_factor, stops, duration) in enumerate(templates, 1):
            price = round(base_price * price_factor, 2)
            # Create simple faux times
            dep_hour = 8 + (idx * 2)
            arr_hour = dep_hour + int(duration)
            ret_dep = 10 + (idx * 2)
            ret_arr = ret_dep + int(duration) + 1
            flights.append({
                "flight_id": f"MOCK-{airline[:3].upper()}-{idx}",
                "airline": airline,
                "airline_logo": logo,
                "price": price,
                "total_price": price,
                "data_source": "mock",
                "fallback_reason": fallback_reason,
                "travel_class": comfort_level.title(),
                "carbon_emissions": random.randint(420, 910),
                "booking_url": "",
                "outbound": {
                    "date": start_date,
                    "airline": airline,
                    "flight_number": f"{airline[:2].upper()}{random.randint(100, 999)}",
                    "duration_hours": duration,
                    "stops": stops,
                    "departure": {
                        "airport": origin,
                        "time": f"{dep_hour:02d}:20"
                    },
                    "arrival": {
                        "airport": destination,
                        "time": f"{arr_hour:02d}:05"
                    },
                    "layovers": ["Reykjavik"] if stops else []
                },
                "return": {
                    "date": end_date,
                    "airline": airline,
                    "flight_number": f"{airline[:2].upper()}{random.randint(400, 799)}",
                    "duration_hours": duration + 0.4,
                    "stops": stops,
                    "departure": {
                        "airport": destination,
                        "time": f"{ret_dep:02d}:35"
                    },
                    "arrival": {
                        "airport": origin,
                        "time": f"{ret_arr:02d}:10"
                    },
                    "layovers": ["Dublin"] if stops else []
                }
            })
        
        return flights

    def _generate_mock_hotels(
        self,
        destination: str,
        check_in: str,
        check_out: str,
        guests: int,
        min_rating: float,
        budget: Optional[float]
    ) -> List[Dict[str, Any]]:
        """Basic mock hotel options to keep the experience smooth without keys."""
        logger.info("Generating mock hotels")
        nights = max((datetime.strptime(check_out, "%Y-%m-%d") - datetime.strptime(check_in, "%Y-%m-%d")).days, 1)
        base_night = (budget / nights) if budget else 140
        templates = [
            ("Harbor Lights Boutique", 4.5, 1.15),
            ("Central Courtyard", 4.2, 0.95),
            ("Traveler's Loft", 3.9, 0.75),
        ]
        hotels = []
        for name, rating, factor in templates:
            per_night = round(base_night * factor, 2)
            hotels.append({
                "name": name,
                "rating": rating,
                "data_source": "mock",
                "type": "Hotel",
                "location": {
                    "address": f"{destination} City Center",
                    "coordinates": {"lat": 0.0, "lng": 0.0}
                },
                "room_type": "Queen with breakfast",
                "price": {
                    "per_night": per_night,
                    "total": round(per_night * nights, 2),
                    "nights": nights,
                    "currency": "USD"
                },
                "amenities": ["WiFi", "Breakfast", "Late Checkout", "Gym"],
                "policies": {"cancellation": "Free cancellation up to 24h before check-in"}
            })
        # Respect minimum rating
        return [h for h in hotels if h["rating"] >= min_rating]

    def _generate_mock_weather(self, destination: str, date: str) -> Dict[str, Any]:
        """Generate mock weather data."""
        conditions = ["Sunny", "Partly Cloudy", "Cloudy", "Rainy", "Stormy", "Clear"]
        
        # Base temperature on month (simple approximation)
        month = int(date.split('-')[1])
        base_temp = 15 + (10 * abs(month - 7) / 7)  # Warmer in summer
        
        return {
            "date": date,
            "location": destination,
            "condition": random.choice(conditions),
            "temperature": {
                "current": round(base_temp + random.uniform(-5, 5), 1),
                "high": round(base_temp + random.uniform(5, 10), 1),
                "low": round(base_temp - random.uniform(0, 5), 1),
                "unit": "°C"
            },
            "humidity": random.randint(40, 90),
            "wind_speed": round(random.uniform(5, 30), 1),
            "precipitation_chance": random.randint(0, 100),
            "uv_index": random.randint(1, 11),
            "sunrise": "06:30 AM",
            "sunset": "18:30 PM",
            "icon": "☀️" if random.choice([True, False]) else "⛅"
        }

    
    async def _fetch_real_weather(self, destination: str, date: str):
        """Fetch real weather data from OpenWeatherMap API."""
        if not self.openweather_key:
            logger.warning("OpenWeatherMap API key not found, using mock data")
            return self._generate_mock_weather(destination, date)
        
        try:
            async with aiohttp.ClientSession() as session:
                # Get coordinates for the destination first
                geo_url = "http://api.openweathermap.org/geo/1.0/direct"
                geo_params = {
                    'q': destination,
                    'limit': 1,
                    'appid': self.openweather_key
                }
                
                async with session.get(geo_url, params=geo_params, timeout=self.api_config['weather']['timeout']) as response:
                    if response.status != 200:
                        logger.warning(f"OpenWeatherMap Geocoding API error {response.status}")
                        return self._generate_mock_weather(destination, date)
                    
                    geo_data = await response.json()
                    if not geo_data:
                        logger.warning("Location not found, using mock data")
                        return self._generate_mock_weather(destination, date)
                    
                    lat = geo_data[0]['lat']
                    lon = geo_data[0]['lon']
                
                # Get weather forecast
                weather_url = "http://api.openweathermap.org/data/2.5/forecast"
                weather_params = {
                    'lat': lat,
                    'lon': lon,
                    'appid': self.openweather_key,
                    'units': 'metric'  # Celsius
                }
                
                async with session.get(weather_url, params=weather_params, timeout=self.api_config['weather']['timeout']) as response:
                    if response.status == 200:
                        data = await response.json()
                        weather = self._parse_openweather_response(data, destination, date)
                        logger.info(f"✅ Fetched real weather for {destination}")
                        return weather
                    else:
                        logger.warning(f"OpenWeatherMap API error {response.status}, using mock data")
                        return self._generate_mock_weather(destination, date)
        
        except Exception as e:
            logger.error(f"Error fetching weather from OpenWeatherMap: {e}")
            return self._generate_mock_weather(destination, date)
    
    def _parse_openweather_response(self, data: Dict, destination: str, date: str) -> Dict:
        """Parse OpenWeatherMap API response into our format."""
        try:
            # Get forecast for the requested date
            target_date = datetime.strptime(date, "%Y-%m-%d").date()
            
            # Find forecast closest to target date
            forecast_data = None
            for item in data.get('list', []):
                forecast_date = datetime.fromtimestamp(item['dt']).date()
                if forecast_date == target_date:
                    forecast_data = item
                    break
            
            # If no exact match, use first forecast
            if not forecast_data and data.get('list'):
                forecast_data = data['list'][0]
            
            if not forecast_data:
                return self._generate_mock_weather(destination, date)
            
            main = forecast_data.get('main', {})
            weather = forecast_data.get('weather', [{}])[0]
            wind = forecast_data.get('wind', {})
            
            # Map weather condition to emoji
            condition_icons = {
                'Clear': '☀️',
                'Clouds': '⛅',
                'Rain': '🌧️',
                'Drizzle': '🌦️',
                'Thunderstorm': '⛈️',
                'Snow': '❄️',
                'Mist': '🌫️',
                'Fog': '🌫️'
            }
            
            return {
                "date": date,
                "location": destination,
                "condition": weather.get('main', 'Clear'),
                "description": weather.get('description', 'clear sky'),
                "temperature": {
                    "current": round(main.get('temp', 20), 1),
                    "high": round(main.get('temp_max', 25), 1),
                    "low": round(main.get('temp_min', 15), 1),
                    "feels_like": round(main.get('feels_like', 20), 1),
                    "unit": "°C"
                },
                "humidity": main.get('humidity', 50),
                "wind_speed": round(wind.get('speed', 5), 1),
                "precipitation_chance": int(forecast_data.get('pop', 0) * 100),
                "uv_index": random.randint(1, 11),  # Not provided by free tier
                "sunrise": "06:30 AM",  # Would need current weather endpoint
                "sunset": "18:30 PM",
                "icon": condition_icons.get(weather.get('main'), '🌤️')
            }
        
        except Exception as e:
            logger.error(f"Error parsing weather data: {e}")
            return self._generate_mock_weather(destination, date)
    
    async def fetch_activities(
        self,
        destination: str,
        preferences: List[str] = None,
        limit: int = 10
    ) -> List[Dict[str, Any]]:
        """
        Fetch activities and points of interest.
        
        Args:
            destination: Destination city
            preferences: User preferences (adventure, culture, etc.)
            limit: Maximum number of activities
            
        Returns:
            List of activities
        """
        logger.info(f"Fetching activities in {destination}")
        
        if self.api_config.get('activities', {}).get('enabled', False):
            logger.debug("Using real activities API")
            return await self._fetch_real_activities(destination, preferences, limit)
        else:
            logger.debug("Using mock activities data")
            return self._generate_mock_activities(destination, preferences, limit)
    
    async def _fetch_real_activities(
        self,
        destination: str,
        preferences: List[str],
        limit: int
    ):
        """Fetch real activities from OSM/Nominatim, Yelp, or other APIs."""
        provider = self.api_config.get('activities', {}).get('provider', 'osm_nominatim')
        
        if provider == 'osm_nominatim':
            return await self._fetch_osm_nominatim_activities(destination, preferences, limit)
        elif provider == 'yelp' and self.yelp_key:
            return await self._fetch_yelp_activities(destination, preferences, limit)
        elif provider == 'opentripmap':
            return await self._fetch_opentripmap_activities(destination, preferences, limit)
        elif provider == 'foursquare' and self.foursquare_key:
            return await self._fetch_foursquare_activities(destination, preferences, limit)
        else:
            logger.warning(f"Activities provider '{provider}' not configured, using mock data")
            return self._generate_mock_activities(destination, preferences, limit)
    
    async def _fetch_osm_nominatim_activities(
        self,
        destination: str,
        preferences: List[str],
        limit: int
    ):
        """
        Fetch activities from OpenStreetMap/Nominatim + Wikipedia.
        100% FREE - NO API KEY REQUIRED!
        """
        try:
            async with aiohttp.ClientSession() as session:
                user_agent = self.api_config.get('activities', {}).get('osm_nominatim', {}).get('user_agent', 'Travelopedia/1.0')
                headers = {'User-Agent': user_agent}
                
                # Step 1: Geocode the destination
                geocode_url = "https://nominatim.openstreetmap.org/search"
                geocode_params = {
                    'q': destination,
                    'format': 'json',
                    'limit': 1
                }
                
                await asyncio.sleep(1)  # Rate limit: 1 req/sec
                
                async with session.get(geocode_url, params=geocode_params, headers=headers, timeout=10) as response:
                    if response.status != 200:
                        logger.warning(f"Nominatim geocoding error {response.status}, using mock data")
                        return self._generate_mock_activities(destination, preferences, limit)
                    
                    geo_data = await response.json()
                    if not geo_data:
                        logger.warning("Could not geocode destination, using mock data")
                        return self._generate_mock_activities(destination, preferences, limit)
                    
                    lat = float(geo_data[0]['lat'])
                    lon = float(geo_data[0]['lon'])
                
                # Step 2: Search for POIs using Overpass API (OpenStreetMap)
                # Map preferences to OSM tags
                tag_mapping = {
                    'adventure': ['sport', 'climbing', 'diving_centre'],
                    'cultural': ['museum', 'theatre', 'art_gallery', 'historic', 'monument'],
                    'culinary': ['restaurant', 'cafe', 'bar'],
                    'nature': ['park', 'garden', 'beach', 'viewpoint', 'nature_reserve'],
                    'relaxation': ['spa', 'wellness_centre'],
                    'family_friendly': ['zoo', 'aquarium', 'theme_park', 'playground']
                }
                
                # Build list of tags based on preferences
                search_tags = []
                if preferences:
                    for pref in preferences:
                        if pref.lower() in tag_mapping:
                            search_tags.extend(tag_mapping[pref.lower()])
                
                # Default tags if none specified
                if not search_tags:
                    search_tags = ['tourism', 'museum', 'monument', 'park', 'viewpoint']
                
                # Remove duplicates
                search_tags = list(set(search_tags))[:5]  # Limit to 5 tags
                
                # Use Overpass API to search for POIs
                radius = self.api_config.get('activities', {}).get('osm_nominatim', {}).get('radius', 10000)
                
                # Build Overpass QL query
                tag_queries = []
                for tag in search_tags:
                    tag_queries.append(f'node["tourism"="{tag}"]({lat-0.1},{lon-0.1},{lat+0.1},{lon+0.1});')
                    tag_queries.append(f'node["amenity"="{tag}"]({lat-0.1},{lon-0.1},{lat+0.1},{lon+0.1});')
                    tag_queries.append(f'node["leisure"="{tag}"]({lat-0.1},{lon-0.1},{lat+0.1},{lon+0.1});')
                
                overpass_query = f"""
                [out:json][timeout:25];
                (
                  node["tourism"]({lat-0.05},{lon-0.05},{lat+0.05},{lon+0.05});
                  node["amenity"~"museum|theatre|arts_centre|cinema"]({lat-0.05},{lon-0.05},{lat+0.05},{lon+0.05});
                  node["leisure"~"park|garden|beach_resort"]({lat-0.05},{lon-0.05},{lat+0.05},{lon+0.05});
                  node["historic"]({lat-0.05},{lon-0.05},{lat+0.05},{lon+0.05});
                );
                out body;
                >;
                out skel qt;
                """
                
                await asyncio.sleep(1)  # Rate limit
                
                overpass_url = "https://overpass-api.de/api/interpreter"
                async with session.post(overpass_url, data={'data': overpass_query}, headers=headers, timeout=30) as response:
                    if response.status != 200:
                        logger.warning(f"Overpass API error {response.status}, using mock data")
                        return self._generate_mock_activities(destination, preferences, limit)
                    
                    osm_data = await response.json()
                    
                    # Parse OSM data
                    activities = []
                    for element in osm_data.get('elements', [])[:limit * 2]:
                        activity = self._parse_osm_element(element, preferences, destination)
                        if activity:
                            activities.append(activity)
                        
                        if len(activities) >= limit:
                            break
                    
                    if activities:
                        logger.info(f"✅ Fetched {len(activities)} real activities from OpenStreetMap (100% FREE!)")
                        return activities
                    else:
                        logger.warning("No activities found in OSM response, using mock data")
                        return self._generate_mock_activities(destination, preferences, limit)
        
        except Exception as e:
            logger.error(f"Error fetching activities from OSM/Nominatim: {e}")
            return self._generate_mock_activities(destination, preferences, limit)
    
    def _parse_osm_element(self, element: Dict, preferences: List[str], destination: str) -> Dict:
        """Parse OpenStreetMap element into our activity format."""
        try:
            tags = element.get('tags', {})
            
            # Get name
            name = tags.get('name', tags.get('name:en', ''))
            if not name:
                return None
            
            # Determine category and type
            tourism_type = tags.get('tourism', '')
            amenity_type = tags.get('amenity', '')
            leisure_type = tags.get('leisure', '')
            historic_type = tags.get('historic', '')
            
            # Map to our categories
            category = 'cultural'
            activity_type = 'Attraction'
            
            if tourism_type in ['museum', 'gallery', 'artwork']:
                category = 'cultural'
                activity_type = 'Museum/Gallery'
            elif tourism_type in ['viewpoint', 'attraction']:
                category = 'nature'
                activity_type = 'Viewpoint'
            elif tourism_type == 'hotel' or amenity_type == 'restaurant':
                return None  # Skip hotels/restaurants
            elif leisure_type in ['park', 'garden', 'beach_resort']:
                category = 'nature'
                activity_type = 'Park/Garden'
            elif historic_type:
                category = 'cultural'
                activity_type = 'Historic Site'
            elif amenity_type in ['theatre', 'cinema', 'arts_centre']:
                category = 'cultural'
                activity_type = 'Entertainment'
            
            # Get location
            lat = element.get('lat', 0)
            lon = element.get('lon', 0)
            
            # Build description
            description = f"{activity_type} in {destination}."
            if tags.get('description'):
                description = tags['description']
            elif tags.get('wikipedia'):
                description = f"Historic {activity_type.lower()} with Wikipedia entry."
            
            # Estimate rating (OSM doesn't have ratings)
            rating = round(random.uniform(4.0, 4.8), 1)
            
            # Get address
            address = tags.get('addr:full', '')
            if not address:
                street = tags.get('addr:street', '')
                housenumber = tags.get('addr:housenumber', '')
                city = tags.get('addr:city', destination)
                if street:
                    address = f"{housenumber} {street}, {city}".strip()
                else:
                    address = city
            
            # Duration based on type
            duration_map = {
                'Museum/Gallery': 2,
                'Historic Site': 2,
                'Park/Garden': 3,
                'Viewpoint': 1,
                'Entertainment': 3
            }
            
            return {
                "activity_id": f"OSM{element.get('id', random.randint(1000, 9999))}",
                "name": name,
                "type": activity_type,
                "category": category,
                "description": description,
                "location": {
                    "address": address,
                    "coordinates": {
                        "lat": lat,
                        "lng": lon
                    }
                },
                "rating": rating,
                "price_level": 1 if tags.get('fee') == 'no' else 2,
                "price": 0 if tags.get('fee') == 'no' else random.choice([10, 15, 20, 25]),  # Estimated entry fee
                "duration_hours": duration_map.get(activity_type, 2),
                "best_time": random.choice(['Morning', 'Afternoon', 'Full Day']),
                "tags": [tourism_type, amenity_type, leisure_type, historic_type] if tourism_type or amenity_type or leisure_type or historic_type else [],
                "website": tags.get('website', tags.get('url', '')),
                "wikipedia": tags.get('wikipedia', ''),
                "photos": []
            }
        
        except Exception as e:
            logger.warning(f"Error parsing OSM element: {e}")
            return None
    
    async def _fetch_yelp_activities(
        self,
        destination: str,
        preferences: List[str],
        limit: int
    ):
        """Fetch activities from Yelp Fusion API (FREE 5,000 calls/day!)."""
        if not self.yelp_key:
            logger.warning("Yelp API key not found, using mock data")
            return self._generate_mock_activities(destination, preferences, limit)
        
        try:
            async with aiohttp.ClientSession() as session:
                # Yelp Fusion API v3
                url = "https://api.yelp.com/v3/businesses/search"
                headers = {
                    "Authorization": f"Bearer {self.yelp_key}",
                    "Accept": "application/json"
                }
                
                # Map preferences to Yelp categories
                category_mapping = {
                    'adventure': 'active,hiking,climbing,diving',
                    'cultural': 'arts,museums,galleries,theater,tours',
                    'culinary': 'restaurants,food,cafes,bars',
                    'nature': 'parks,hiking,beaches',
                    'relaxation': 'spas,massage,wellness',
                    'family_friendly': 'zoos,amusementparks,aquariums',
                    'nightlife': 'nightlife,bars,clubs',
                    'shopping': 'shopping'
                }
                
                # Build categories filter
                categories = []
                if preferences:
                    for pref in preferences:
                        if pref.lower() in category_mapping:
                            categories.extend(category_mapping[pref.lower()].split(','))
                
                # Default categories if none specified
                if not categories:
                    categories = ['arts', 'active', 'food', 'tours']
                
                # Remove duplicates
                categories = list(set(categories))[:3]  # Yelp allows max 3 categories
                
                # Get config values
                radius = self.api_config.get('activities', {}).get('yelp', {}).get('radius', 10000)
                
                params = {
                    'location': destination,
                    'categories': ','.join(categories),
                    'limit': min(limit, 50),  # Max 50 per request
                    'radius': min(radius, 40000),  # Max 40km
                    'sort_by': 'rating'  # Sort by rating
                }
                
                async with session.get(url, headers=headers, params=params, timeout=self.api_config.get('activities', {}).get('timeout', 15)) as response:
                    if response.status == 200:
                        data = await response.json()
                        activities = self._parse_yelp_response(data, preferences)
                        
                        if activities:
                            logger.info(f"✅ Fetched {len(activities)} real activities from Yelp Fusion API (FREE!)")
                            return activities
                        else:
                            logger.warning("No activities found in Yelp response, using mock data")
                            return self._generate_mock_activities(destination, preferences, limit)
                    elif response.status == 401:
                        error_text = await response.text()
                        logger.warning(f"Yelp API 401 Unauthorized: {error_text[:200]}")
                        logger.warning("Please check your YELP_API_KEY in .env file")
                        return self._generate_mock_activities(destination, preferences, limit)
                    else:
                        error_text = await response.text()
                        logger.warning(f"Yelp API error {response.status}: {error_text[:200]}, using mock data")
                        return self._generate_mock_activities(destination, preferences, limit)
        
        except Exception as e:
            logger.error(f"Error fetching activities from Yelp: {e}")
            return self._generate_mock_activities(destination, preferences, limit)
    
    def _parse_yelp_response(self, data: Dict, preferences: List[str]) -> List[Dict]:
        """Parse Yelp Fusion API response into our format."""
        activities = []
        
        if 'businesses' not in data:
            return []
        
        for business in data['businesses']:
            try:
                # Extract categories
                categories = business.get('categories', [])
                category_names = [cat.get('title', '') for cat in categories]
                
                # Determine activity type from categories
                activity_type = category_names[0] if category_names else 'Attraction'
                
                # Map to our preference categories
                activity_category = 'cultural'
                category_str = ' '.join(category_names).lower()
                
                if any(word in category_str for word in ['park', 'outdoor', 'hiking', 'beach', 'nature']):
                    activity_category = 'nature'
                elif any(word in category_str for word in ['restaurant', 'food', 'cafe', 'bar', 'dining']):
                    activity_category = 'culinary'
                elif any(word in category_str for word in ['museum', 'art', 'gallery', 'theater', 'cultural', 'tour']):
                    activity_category = 'cultural'
                elif any(word in category_str for word in ['spa', 'wellness', 'massage', 'yoga']):
                    activity_category = 'relaxation'
                elif any(word in category_str for word in ['active', 'sport', 'climbing', 'diving', 'adventure']):
                    activity_category = 'adventure'
                elif any(word in category_str for word in ['nightlife', 'club', 'bar']):
                    activity_category = 'nightlife'
                
                # Extract location
                location_data = business.get('location', {})
                coordinates = business.get('coordinates', {})
                
                # Extract rating
                rating = business.get('rating', 4.0)
                
                # Extract price level ($ to $$$$)
                price_str = business.get('price', '$$')
                price_level = len(price_str) if price_str else 2
                
                # Estimate duration based on category
                duration_map = {
                    'nature': random.randint(2, 4),
                    'cultural': random.randint(1, 3),
                    'adventure': random.randint(2, 5),
                    'culinary': random.randint(1, 2),
                    'relaxation': random.randint(1, 3),
                    'nightlife': random.randint(2, 4)
                }
                
                # Build address string
                address_parts = [
                    location_data.get('address1', ''),
                    location_data.get('city', ''),
                    location_data.get('state', '')
                ]
                address = ', '.join([part for part in address_parts if part])
                
                activities.append({
                    "activity_id": business.get('id', f"YLP{random.randint(1000, 9999)}"),
                    "name": business.get('name', 'Unknown Activity'),
                    "type": activity_type,
                    "category": activity_category,
                    "description": f"{activity_type} in {location_data.get('city', 'the area')}. " + 
                                  (f"Review: {business.get('review_count', 0)} reviews" if business.get('review_count') else ''),
                    "location": {
                        "address": address,
                        "coordinates": {
                            "lat": coordinates.get('latitude', 0),
                            "lng": coordinates.get('longitude', 0)
                        }
                    },
                    "rating": rating,
                    "review_count": business.get('review_count', 0),
                    "price_level": price_level,
                    "duration_hours": duration_map.get(activity_category, 2),
                    "best_time": random.choice(['Morning', 'Afternoon', 'Evening', 'Full Day']),
                    "tags": category_names[:3],
                    "phone": business.get('phone', ''),
                    "website": business.get('url', ''),
                    "photos": [business.get('image_url')] if business.get('image_url') else [],
                    "is_closed": business.get('is_closed', False)
                })
            except Exception as e:
                logger.warning(f"Error parsing Yelp business data: {e}")
                continue
        
        return activities if activities else []
    
    async def _fetch_opentripmap_activities(
        self,
        destination: str,
        preferences: List[str],
        limit: int
    ):
        """Fetch activities from OpenTripMap API (NO API KEY NEEDED!)."""
        try:
            async with aiohttp.ClientSession() as session:
                # Step 1: Geocode the destination
                geocode_url = "https://api.opentripmap.com/0.1/en/places/geoname"
                params = {'name': destination}
                
                async with session.get(geocode_url, params=params, timeout=self.api_config.get('activities', {}).get('timeout', 15)) as response:
                    if response.status != 200:
                        logger.warning(f"OpenTripMap geocoding error {response.status}, using mock data")
                        return self._generate_mock_activities(destination, preferences, limit)
                    
                    geo_data = await response.json()
                    if not geo_data or 'lat' not in geo_data:
                        logger.warning("OpenTripMap: Could not geocode destination, using mock data")
                        return self._generate_mock_activities(destination, preferences, limit)
                    
                    lat = geo_data['lat']
                    lon = geo_data['lon']
                
                # Step 2: Search for places of interest
                # Map preferences to OpenTripMap kinds
                kinds_mapping = {
                    'adventure': 'natural,sport,natural',
                    'cultural': 'cultural,historic,museums,theatres',
                    'culinary': 'foods',
                    'nature': 'natural,beaches,nature_reserves',
                    'relaxation': 'natural,beaches',
                    'family_friendly': 'amusements,cultural'
                }
                
                # Build kinds parameter based on preferences
                kinds = []
                if preferences:
                    for pref in preferences:
                        if pref.lower() in kinds_mapping:
                            kinds.extend(kinds_mapping[pref.lower()].split(','))
                
                if not kinds:
                    kinds = ['interesting_places', 'cultural', 'natural', 'amusements']
                
                # Remove duplicates
                kinds = ','.join(list(set(kinds)))
                
                radius = self.api_config.get('activities', {}).get('opentripmap', {}).get('radius', 10000)
                
                places_url = "https://api.opentripmap.com/0.1/en/places/radius"
                places_params = {
                    'radius': radius,
                    'lon': lon,
                    'lat': lat,
                    'kinds': kinds,
                    'limit': min(limit * 2, 50)  # Get more to filter
                }
                
                async with session.get(places_url, params=places_params, timeout=self.api_config.get('activities', {}).get('timeout', 15)) as response:
                    if response.status == 200:
                        data = await response.json()
                        
                        # Get detailed info for each place
                        activities = []
                        for place in data[:limit]:
                            xid = place.get('xid')
                            if xid:
                                # Get place details
                                detail_url = f"https://api.opentripmap.com/0.1/en/places/xid/{xid}"
                                async with session.get(detail_url, timeout=10) as detail_response:
                                    if detail_response.status == 200:
                                        detail = await detail_response.json()
                                        activity = self._parse_opentripmap_place(detail, preferences)
                                        if activity:
                                            activities.append(activity)
                                
                                if len(activities) >= limit:
                                    break
                        
                        if activities:
                            logger.info(f"✅ Fetched {len(activities)} real activities from OpenTripMap (FREE!)")
                            return activities
                        else:
                            logger.warning("No activities found in OpenTripMap response, using mock data")
                            return self._generate_mock_activities(destination, preferences, limit)
                    else:
                        logger.warning(f"OpenTripMap API error {response.status}, using mock data")
                        return self._generate_mock_activities(destination, preferences, limit)
        
        except Exception as e:
            logger.error(f"Error fetching activities from OpenTripMap: {e}")
            return self._generate_mock_activities(destination, preferences, limit)
    
    def _parse_opentripmap_place(self, place: Dict, preferences: List[str]) -> Dict:
        """Parse OpenTripMap place data into our format."""
        try:
            name = place.get('name', 'Unknown Place')
            if not name or name == '':
                return None
            
            # Determine category based on kinds
            kinds = place.get('kinds', '').lower()
            category = 'cultural'
            activity_type = 'Attraction'
            
            if 'natural' in kinds or 'beach' in kinds or 'nature' in kinds:
                category = 'nature'
                activity_type = 'Nature'
            elif 'museum' in kinds or 'cultural' in kinds or 'historic' in kinds:
                category = 'cultural'
                activity_type = 'Cultural'
            elif 'sport' in kinds or 'climbing' in kinds:
                category = 'adventure'
                activity_type = 'Adventure'
            elif 'food' in kinds or 'restaurant' in kinds:
                category = 'culinary'
                activity_type = 'Culinary'
            
            # Extract location
            point = place.get('point', {})
            address = place.get('address', {})
            
            # Get rating (OpenTripMap uses "rate" 1-7 scale)
            rate = place.get('rate', 0)
            rating = round(min(rate / 7 * 5, 5.0), 1) if rate else 4.0
            
            # Estimate duration based on category
            duration_map = {
                'nature': random.randint(2, 4),
                'cultural': random.randint(1, 3),
                'adventure': random.randint(2, 5),
                'culinary': random.randint(1, 2)
            }
            
            return {
                "activity_id": place.get('xid', f"OTM{random.randint(1000, 9999)}"),
                "name": name,
                "type": activity_type,
                "category": category,
                "description": place.get('info', {}).get('descr', f"Popular {activity_type.lower()} in the area"),
                "location": {
                    "address": address.get('road', '') + ', ' + address.get('city', ''),
                    "coordinates": {
                        "lat": point.get('lat', 0),
                        "lng": point.get('lon', 0)
                    }
                },
                "rating": rating,
                "price_level": random.randint(1, 3),
                "duration_hours": duration_map.get(category, 2),
                "best_time": random.choice(['Morning', 'Afternoon', 'Evening', 'Full Day']),
                "tags": kinds.split(',')[:3] if kinds else [],
                "website": place.get('url', place.get('wikipedia', '')),
                "photos": [place.get('preview', {}).get('source', '')] if place.get('preview') else []
            }
        
        except Exception as e:
            logger.warning(f"Error parsing OpenTripMap place: {e}")
            return None
    
    async def _fetch_foursquare_activities(
        self,
        destination: str,
        preferences: List[str],
        limit: int
    ):
        """Fetch activities from Foursquare API (if key is provided)."""
        if not self.foursquare_key:
            logger.warning("Foursquare API key not found, using mock data")
            return self._generate_mock_activities(destination, preferences, limit)
        
        try:
            async with aiohttp.ClientSession() as session:
                # NEW Foursquare Places API v3 (2025-06-17 version)
                url = "https://places-api.foursquare.com/places/search"
                headers = {
                    "Authorization": self.foursquare_key,
                    "X-Places-Api-Version": "2025-06-17",  # Required version header
                    "Accept": "application/json"
                }
                
                # Map preferences to Foursquare categories
                category_mapping = {
                    'adventure': '16000',      # Outdoors & Recreation
                    'cultural': '10000',       # Arts & Entertainment
                    'culinary': '13000',       # Food & Drink
                    'nature': '16000',         # Outdoors & Recreation
                    'relaxation': '17000',     # Health & Beauty
                    'family_friendly': '12000' # Community & Government
                }
                
                # Build category filter
                categories = []
                if preferences:
                    for pref in preferences:
                        if pref.lower() in category_mapping:
                            categories.append(category_mapping[pref.lower()])
                
                params = {
                    'near': destination,
                    'limit': min(limit, 50),  # Max 50 per request
                    'sort': 'POPULARITY'
                }
                
                # Add category filter if preferences specified
                if categories:
                    params['fsq_category_ids'] = ','.join(categories)
                
                async with session.get(url, headers=headers, params=params, timeout=self.api_config.get('activities', {}).get('timeout', 15)) as response:
                    if response.status == 200:
                        data = await response.json()
                        activities = self._parse_foursquare_v3_response(data, preferences)
                        
                        if activities:
                            logger.info(f"✅ Fetched {len(activities)} real activities from Foursquare Places API v3")
                            return activities
                        else:
                            logger.warning("No activities found in API response, using mock data")
                            return self._generate_mock_activities(destination, preferences, limit)
                    elif response.status == 401:
                        error_text = await response.text()
                        logger.warning(f"Foursquare API 401 Unauthorized: {error_text[:200]}")
                        logger.warning("Make sure you're using a Service API Key (not Legacy API Key)")
                        return self._generate_mock_activities(destination, preferences, limit)
                    else:
                        error_text = await response.text()
                        logger.warning(f"Foursquare API error {response.status}: {error_text[:200]}, using mock data")
                        return self._generate_mock_activities(destination, preferences, limit)
        
        except Exception as e:
            logger.error(f"Error fetching activities from Foursquare: {e}")
            return self._generate_mock_activities(destination, preferences, limit)
    
    def _parse_foursquare_v3_response(self, data: Dict, preferences: List[str]) -> List[Dict]:
        """Parse Foursquare Places API v3 (2025-06-17) response into our format."""
        activities = []
        
        if 'results' not in data:
            return []
        
        for place in data['results']:
            try:
                # Extract categories
                categories = place.get('categories', [])
                category_names = [cat.get('name', '') for cat in categories]
                
                # Determine activity type from categories
                activity_type = category_names[0] if category_names else 'Attraction'
                
                # Map to our preference categories
                activity_category = 'cultural'
                category_str = ' '.join(category_names).lower()
                if any(word in category_str for word in ['park', 'outdoor', 'nature', 'beach', 'hiking']):
                    activity_category = 'nature'
                elif any(word in category_str for word in ['restaurant', 'food', 'dining', 'cafe', 'bar']):
                    activity_category = 'culinary'
                elif any(word in category_str for word in ['museum', 'art', 'historic', 'cultural', 'theater']):
                    activity_category = 'cultural'
                elif any(word in category_str for word in ['spa', 'wellness', 'relaxation', 'massage']):
                    activity_category = 'relaxation'
                elif any(word in category_str for word in ['adventure', 'sports', 'recreation']):
                    activity_category = 'adventure'
                
                # Extract location
                location_data = place.get('location', {})
                
                # Extract rating (0-10 scale, convert to 5-scale)
                rating = place.get('rating', 0)
                if rating > 0:
                    rating = round(rating / 2, 1)  # Convert 10-scale to 5-scale
                else:
                    rating = round(random.uniform(4.0, 5.0), 1)
                
                # Extract price level
                price_level = place.get('price', random.randint(1, 4))
                
                activities.append({
                    "activity_id": place.get('fsq_place_id', f"ACT{random.randint(1000, 9999)}"),
                    "name": place.get('name', 'Unknown Activity'),
                    "type": activity_type,
                    "category": activity_category,
                    "description": place.get('description', f"Popular {activity_type.lower()} in the area"),
                    "location": {
                        "address": location_data.get('formatted_address', location_data.get('address', 'N/A')),
                        "coordinates": {
                            "lat": place.get('geocodes', {}).get('main', {}).get('latitude', 0),
                            "lng": place.get('geocodes', {}).get('main', {}).get('longitude', 0)
                        }
                    },
                    "rating": rating,
                    "price_level": price_level,
                    "duration_hours": random.randint(1, 4),
                    "best_time": random.choice(['Morning', 'Afternoon', 'Evening', 'Full Day']),
                    "tags": category_names[:3],
                    "website": place.get('website', ''),
                    "photos": [photo.get('prefix', '') + 'original' + photo.get('suffix', '') 
                              for photo in place.get('photos', [])[:3]] if place.get('photos') else []
                })
            except Exception as e:
                logger.warning(f"Error parsing activity data: {e}")
                continue
        
        return activities if activities else []
    
    def _generate_mock_activities(
        self,
        destination: str,
        preferences: List[str],
        limit: int
    ) -> List[Dict]:
        """Generate mock activities data."""
        activity_templates = {
            'adventure': [
                'Zip Lining Adventure', 'Rock Climbing', 'White Water Rafting',
                'Hiking Trail', 'Mountain Biking', 'Paragliding Experience'
            ],
            'cultural': [
                'Historical Museum Visit', 'Art Gallery Tour', 'Cultural Walking Tour',
                'Traditional Dance Performance', 'Heritage Site Visit', 'Local Market Tour'
            ],
            'culinary': [
                'Cooking Class', 'Food Tour', 'Wine Tasting', 'Local Restaurant',
                'Street Food Adventure', 'Farm to Table Experience'
            ],
            'nature': [
                'National Park Visit', 'Botanical Garden', 'Beach Day',
                'Wildlife Safari', 'Scenic Viewpoint', 'Nature Walk'
            ],
            'relaxation': [
                'Spa Day', 'Yoga Session', 'Meditation Retreat',
                'Hot Springs Visit', 'Wellness Center', 'Beach Relaxation'
            ]
        }
        
        activities = []
        activity_pool = []
        
        # Prioritize activities based on preferences
        if preferences:
            for pref in preferences:
                if pref.lower() in activity_templates:
                    activity_pool.extend(activity_templates[pref.lower()])
        
        # Add some random activities
        for category_activities in activity_templates.values():
            activity_pool.extend(category_activities)
        
        # Remove duplicates and limit
        activity_pool = list(set(activity_pool))
        random.shuffle(activity_pool)
        
        for i, activity_name in enumerate(activity_pool[:limit]):
            category = next((cat for cat, acts in activity_templates.items() if activity_name in acts), 'cultural')
            
            activities.append({
                "activity_id": f"ACT{random.randint(1000, 9999)}",
                "name": f"{activity_name} in {destination}",
                "type": category.title(),
                "category": category,
                "description": f"Experience the best {activity_name.lower()} that {destination} has to offer.",
                "location": {
                    "address": f"{destination} City Center",
                    "coordinates": {
                        "lat": round(random.uniform(-90, 90), 6),
                        "lng": round(random.uniform(-180, 180), 6)
                    }
                },
                "rating": round(random.uniform(4.0, 5.0), 1),
                "price_level": random.randint(1, 4),
                "duration_hours": random.randint(1, 6),
                "best_time": random.choice(['Morning', 'Afternoon', 'Evening', 'Full Day']),
                "tags": [category, destination, random.choice(['Popular', 'Recommended', 'Hidden Gem'])],
                "website": f"https://example.com/activity/{i}",
                "photos": []
            })
        
        return activities
