"""
Airport code converter for common cities.
Maps city names to their primary airport IATA codes.
"""

TYPO_CODES = {
    "jsk": "JFK",  # common typo
}

AIRPORT_CODES = {
    # North America
    "new york": "JFK",
    "new york, usa": "JFK",
    "new york city": "JFK",
    "nyc": "JFK",
    "los angeles": "LAX",
    "los angeles, usa": "LAX",
    "la": "LAX",
    "chicago": "ORD",
    "chicago, usa": "ORD",
    "san francisco": "SFO",
    "san francisco, usa": "SFO",
    "miami": "MIA",
    "miami, usa": "MIA",
    "boston": "BOS",
    "boston, usa": "BOS",
    "seattle": "SEA",
    "seattle, usa": "SEA",
    "washington": "IAD",
    "washington dc": "IAD",
    "washington, usa": "IAD",
    "toronto": "YYZ",
    "toronto, canada": "YYZ",
    "vancouver": "YVR",
    "vancouver, canada": "YVR",
    "mexico city": "MEX",
    "mexico city, mexico": "MEX",
    
    # Europe
    "london": "LHR",
    "london, uk": "LHR",
    "london, england": "LHR",
    "paris": "CDG",
    "paris, france": "CDG",
    "rome": "FCO",
    "rome, italy": "FCO",
    "madrid": "MAD",
    "madrid, spain": "MAD",
    "barcelona": "BCN",
    "barcelona, spain": "BCN",
    "amsterdam": "AMS",
    "amsterdam, netherlands": "AMS",
    "berlin": "BER",
    "berlin, germany": "BER",
    "frankfurt": "FRA",
    "frankfurt, germany": "FRA",
    "munich": "MUC",
    "munich, germany": "MUC",
    "zurich": "ZRH",
    "zurich, switzerland": "ZRH",
    "vienna": "VIE",
    "vienna, austria": "VIE",
    "athens": "ATH",
    "athens, greece": "ATH",
    "istanbul": "IST",
    "istanbul, turkey": "IST",
    "dubai": "DXB",
    "dubai, uae": "DXB",
    "abu dhabi": "AUH",
    "abu dhabi, uae": "AUH",
    
    # Asia
    "tokyo": "NRT",
    "tokyo, japan": "NRT",
    "osaka": "KIX",
    "osaka, japan": "KIX",
    "singapore": "SIN",
    "singapore, singapore": "SIN",
    "hong kong": "HKG",
    "hong kong, china": "HKG",
    "beijing": "PEK",
    "beijing, china": "PEK",
    "shanghai": "PVG",
    "shanghai, china": "PVG",
    "seoul": "ICN",
    "seoul, south korea": "ICN",
    "bangkok": "BKK",
    "bangkok, thailand": "BKK",
    "kuala lumpur": "KUL",
    "kuala lumpur, malaysia": "KUL",
    "bali": "DPS",
    "bali, indonesia": "DPS",
    "denpasar": "DPS",
    "denpasar, indonesia": "DPS",
    "jakarta": "CGK",
    "jakarta, indonesia": "CGK",
    "manila": "MNL",
    "manila, philippines": "MNL",
    "delhi": "DEL",
    "delhi, india": "DEL",
    "new delhi": "DEL",
    "new delhi, india": "DEL",
    "mumbai": "BOM",
    "mumbai, india": "BOM",
    "bangalore": "BLR",
    "bangalore, india": "BLR",
    "chennai": "MAA",
    "chennai, india": "MAA",
    "kolkata": "CCU",
    "kolkata, india": "CCU",
    "goa": "GOI",
    "goa, india": "GOI",
    "hyderabad": "HYD",
    "hyderabad, india": "HYD",
    "ahmedabad": "AMD",
    "ahmedabad, india": "AMD",
    
    # Oceania
    "sydney": "SYD",
    "sydney, australia": "SYD",
    "melbourne": "MEL",
    "melbourne, australia": "MEL",
    "brisbane": "BNE",
    "brisbane, australia": "BNE",
    "auckland": "AKL",
    "auckland, new zealand": "AKL",
    
    # South America
    "sao paulo": "GRU",
    "sao paulo, brazil": "GRU",
    "rio de janeiro": "GIG",
    "rio de janeiro, brazil": "GIG",
    "buenos aires": "EZE",
    "buenos aires, argentina": "EZE",
    "lima": "LIM",
    "lima, peru": "LIM",
    "bogota": "BOG",
    "bogota, colombia": "BOG",
    
    # Africa
    "johannesburg": "JNB",
    "johannesburg, south africa": "JNB",
    "cape town": "CPT",
    "cape town, south africa": "CPT",
    "cairo": "CAI",
    "cairo, egypt": "CAI",
    "nairobi": "NBO",
    "nairobi, kenya": "NBO",
    "lagos": "LOS",
    "lagos, nigeria": "LOS",
}


def get_airport_code(city_name: str) -> str:
    """
    Convert city name to airport code.
    
    Args:
        city_name: City name (e.g., "New York, USA", "Bali, Indonesia")
    
    Returns:
        Airport IATA code (e.g., "JFK", "DPS")
        Returns original if no mapping found
    """
    # Normalize the city name
    normalized = city_name.lower().strip()
    
    # Handle known typos to nearest major airport
    if normalized in TYPO_CODES:
        return TYPO_CODES[normalized]
    
    # Check if it's already an airport code (3 letters, uppercase)
    if len(city_name) == 3 and city_name.isupper():
        return city_name
    
    # Look up in dictionary
    if normalized in AIRPORT_CODES:
        return AIRPORT_CODES[normalized]
    
    # If user typed "City, COUNTRYCODE", try matching just the city part
    if ',' in normalized:
        city_only = normalized.split(',')[0].strip()
        if city_only in AIRPORT_CODES:
            return AIRPORT_CODES[city_only]
    
    # No mapping found - return original
    return city_name


def suggest_airport_codes(query: str) -> list:
    """
    Suggest matching airport codes for a query.
    
    Args:
        query: Partial city name
    
    Returns:
        List of (city_name, airport_code) tuples
    """
    query = query.lower().strip()
    matches = []
    
    for city, code in AIRPORT_CODES.items():
        if query in city:
            matches.append((city.title(), code))
    
    return matches[:10]  # Return top 10 matches


def format_city_airport(city_name: str, airport_code: str) -> str:
    """
    Format city name with airport code.
    
    Args:
        city_name: City name
        airport_code: Airport code
    
    Returns:
        Formatted string "City Name (CODE)"
    """
    return f"{city_name} ({airport_code})"
