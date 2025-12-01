"""
Budget Optimizer for cost vs. comfort tradeoff analysis.
Generates multiple itinerary alternatives with different budget allocations.
"""

from typing import Dict, List, Any, Tuple
import yaml
from backend.utils.logger import get_logger

logger = get_logger(__name__)


class BudgetOptimizer:
    """
    Optimizes travel budget allocation across different categories.
    Performs Pareto optimization for cost vs. comfort tradeoffs.
    """
    
    def __init__(self, config_path: str = "backend/utils/config.yaml"):
        """Initialize budget optimizer."""
        with open(config_path, 'r') as f:
            self.config = yaml.safe_load(f)
        
        self.budget_config = self.config['budget']
        self.comfort_levels = self.budget_config['comfort_levels']
        self.default_allocation = self.budget_config['default_allocation']
        
        logger.info("Budget Optimizer initialized")
    
    def optimize_budget(
        self,
        total_budget: float,
        duration_days: int,
        flights: List[Dict[str, Any]],
        hotels: List[Dict[str, Any]],
        activities: List[Dict[str, Any]],
        preferences: Dict[str, Any]
    ) -> Dict[str, Any]:
        """
        Optimize budget allocation across travel components.
        
        Args:
            total_budget: Total available budget
            duration_days: Trip duration in days
            flights: Available flight options
            hotels: Available hotel options
            activities: Available activity options
            preferences: User preferences
            
        Returns:
            Optimized budget allocation with recommendations
        """
        logger.info(f"Optimizing ${total_budget} budget for {duration_days} days")
        
        comfort_level = preferences.get('comfort_level', 'standard')
        
        # Calculate base allocation
        allocation = self._calculate_allocation(
            total_budget,
            comfort_level,
            duration_days
        )
        
        logger.info(f"📊 Budget allocation: Transport=${allocation['transport']:.2f}, Accommodation=${allocation['accommodation']:.2f}, Activities=${allocation['activities']:.2f}")
        logger.info(f"📦 Available options: {len(flights)} flights, {len(hotels)} hotels, {len(activities)} activities")
        
        # Select best options within budget
        selected_flight = self._select_flight(flights, allocation['transport'])
        logger.info(f"✈️  Selected flight: {selected_flight['airline'] if selected_flight else 'None'} - ${selected_flight['price'] if selected_flight else 0}")
        
        selected_hotel = self._select_hotel(hotels, allocation['accommodation'], duration_days)
        logger.info(f"🏨 Selected hotel: {selected_hotel['name'] if selected_hotel else 'None'} - ${selected_hotel['price']['total'] if selected_hotel else 0}")
        
        # Include activities using allocated budget (up to 10)
        selected_activities = self._select_activities(activities, allocation['activities'], duration_days, max_activities=10) if activities else []
        logger.info(f"🎯 Selected {len(selected_activities)} activities totaling ${sum(a.get('price',0) for a in selected_activities)}")
        
        # Calculate actual costs
        actual_costs = {
            'transport': selected_flight['price'] if selected_flight else 0,
            'accommodation': selected_hotel['price']['total'] if selected_hotel else 0,
            'activities': sum(a.get('price', 0) for a in selected_activities),
            'food': allocation.get('food', 0),
            'miscellaneous': allocation.get('miscellaneous', 0)
        }
        
        total_cost = sum(actual_costs.values())
        
        # Calculate savings/overage
        balance = total_budget - total_cost
        
        result = {
            'total_budget': total_budget,
            'total_cost': total_cost,
            'balance': balance,
            'allocation': allocation,
            'actual_costs': actual_costs,
            'selected_options': {
                'flight': selected_flight,
                'hotel': selected_hotel,
                'activities': selected_activities
            },
            'comfort_level': comfort_level,
            'value_score': self._calculate_value_score(
                actual_costs,
                selected_flight,
                selected_hotel,
                selected_activities
            )
        }
        
        logger.info(f"Optimization complete - Balance: ${balance:.2f}, Value score: {result['value_score']:.2f}")
        return result
    
    def generate_alternatives(
        self,
        total_budget: float,
        duration_days: int,
        flights: List[Dict[str, Any]],
        hotels: List[Dict[str, Any]],
        activities: List[Dict[str, Any]],
        preferences: Dict[str, Any]
    ) -> List[Dict[str, Any]]:
        """
        Generate multiple budget alternatives with different comfort levels.
        
        Args:
            total_budget: Total available budget
            duration_days: Trip duration
            flights: Available flights
            hotels: Available hotels
            activities: Available activities
            preferences: User preferences
            
        Returns:
            List of alternative budget plans
        """
        logger.info("Generating budget alternatives")
        
        alternatives = []
        comfort_levels = ['budget', 'standard', 'comfort']
        
        for comfort in comfort_levels:
            # Adjust preferences for this comfort level
            adjusted_prefs = preferences.copy()
            adjusted_prefs['comfort_level'] = comfort
            
            # Optimize for this comfort level
            alt = self.optimize_budget(
                total_budget,
                duration_days,
                flights,
                hotels,
                activities,
                adjusted_prefs
            )
            
            alt['label'] = comfort.title()
            alt['description'] = self._get_comfort_description(comfort)
            
            alternatives.append(alt)
        
        # Sort by value score
        alternatives.sort(key=lambda x: x['value_score'], reverse=True)
        
        logger.info(f"Generated {len(alternatives)} alternatives")
        return alternatives
    
    def _calculate_allocation(
        self,
        total_budget: float,
        comfort_level: str,
        duration_days: int
    ) -> Dict[str, float]:
        """Calculate budget allocation across categories."""
        multiplier = self.comfort_levels.get(comfort_level, 1.0)
        
        allocation = {}
        for category, percentage in self.default_allocation.items():
            base_amount = total_budget * percentage
            
            # Adjust for comfort level
            if category in ['accommodation', 'transport']:
                allocation[category] = base_amount * multiplier
            else:
                allocation[category] = base_amount
        
        # Normalize to ensure sum equals total budget
        total_allocated = sum(allocation.values())
        if total_allocated > 0:
            factor = total_budget / total_allocated
            allocation = {k: v * factor for k, v in allocation.items()}
        
        return allocation
    
    def _select_flight(
        self,
        flights: List[Dict[str, Any]],
        budget: float
    ) -> Dict[str, Any]:
        """Select best flight within budget."""
        # Filter flights within budget
        affordable = [f for f in flights if f['price'] <= budget]
        
        if not affordable:
            # Return cheapest if nothing affordable
            return min(flights, key=lambda x: x['price']) if flights else None
        
        # Score flights by value (considering price, duration, stops)
        for flight in affordable:
            stops = flight.get('outbound', {}).get('stops', 0)
            price_ratio = flight['price'] / budget
            
            # Lower is better for both
            flight['value_score'] = price_ratio + (stops * 0.1)
        
        # Return flight with best value score
        return min(affordable, key=lambda x: x['value_score'])
    
    def _select_hotel(
        self,
        hotels: List[Dict[str, Any]],
        budget: float,
        nights: int
    ) -> Dict[str, Any]:
        """Select best hotel within budget."""
        # Filter hotels within budget
        affordable = [h for h in hotels if h['price']['total'] <= budget]
        
        if not affordable:
            # Return cheapest if nothing affordable
            return min(hotels, key=lambda x: x['price']['total']) if hotels else None
        
        # Score hotels by value (rating vs price)
        for hotel in affordable:
            rating = hotel.get('rating', 3.5)
            # Avoid division by zero; treat zero/near-zero budget as very small denominator
            price_ratio = hotel['price']['total'] / budget if budget else float('inf')
            if price_ratio == 0:
                price_ratio = 0.0001
            
            # Higher rating and lower price ratio is better
            hotel['value_score'] = rating / price_ratio
        
        # Return hotel with best value score
        return max(affordable, key=lambda x: x['value_score'])
    
    def _select_activities(
        self,
        activities: List[Dict[str, Any]],
        budget: float,
        duration_days: int,
        max_activities: int = None
    ) -> List[Dict[str, Any]]:
        """Select activities within budget."""
        # Sort activities by personalization score
        sorted_activities = sorted(
            activities,
            key=lambda x: x.get('personalization_score', 0),
            reverse=True
        )
        
        selected = []
        total_cost = 0
        max_activities = max_activities or (duration_days * 2)  # ~2 activities per day
        
        for activity in sorted_activities:
            if len(selected) >= max_activities:
                break
            # Handle missing or malformed price gracefully
            price_val = activity.get('price', 0) or 0
            try:
                price_val = float(price_val)
            except Exception:
                price_val = 0
            
            if total_cost + price_val <= budget:
                selected.append(activity)
                total_cost += price_val
        
        return selected
    
    def _calculate_value_score(
        self,
        costs: Dict[str, float],
        flight: Dict[str, Any],
        hotel: Dict[str, Any],
        activities: List[Dict[str, Any]]
    ) -> float:
        """Calculate overall value score for the itinerary."""
        score = 0
        
        # Flight value (lower stops = higher value)
        if flight:
            stops = flight.get('outbound', {}).get('stops', 0)
            score += (3 - stops) / 3 * 20  # Max 20 points
        
        # Hotel value (rating-based)
        if hotel:
            rating = hotel.get('rating', 3.5)
            score += (rating / 5.0) * 30  # Max 30 points
        
        # Activity value (count and personalization)
        if activities:
            avg_pers_score = sum(a.get('personalization_score', 0.5) for a in activities) / len(activities)
            activity_count_score = min(len(activities) / 10, 1.0)  # Normalize to max 10 activities
            score += (avg_pers_score * 25) + (activity_count_score * 25)  # Max 50 points
        
        return round(score, 2)
    
    def _get_comfort_description(self, comfort_level: str) -> str:
        """Get description for comfort level."""
        descriptions = {
            'budget': 'Maximize savings with essential amenities',
            'standard': 'Balance between cost and comfort',
            'comfort': 'Premium experience with better amenities',
            'luxury': 'Highest quality with luxury services'
        }
        return descriptions.get(comfort_level, '')
    
    def analyze_budget_feasibility(
        self,
        total_budget: float,
        destination: str,
        duration_days: int,
        group_size: int
    ) -> Dict[str, Any]:
        """
        Analyze if the budget is feasible for the trip.
        
        Args:
            total_budget: Total budget
            destination: Destination city
            duration_days: Trip duration
            group_size: Number of travelers
            
        Returns:
            Feasibility analysis
        """
        logger.info("Analyzing budget feasibility")
        
        # Per-person budget
        per_person_budget = total_budget / group_size
        daily_budget = per_person_budget / duration_days
        
        # Rough estimates for minimum costs
        min_flight = 200  # Minimum round-trip flight
        min_hotel_per_night = 50
        min_daily_food = 30
        min_daily_misc = 20
        
        min_required = (
            (min_flight * group_size) +
            (min_hotel_per_night * duration_days) +
            ((min_daily_food + min_daily_misc) * duration_days * group_size)
        )
        
        feasible = total_budget >= min_required
        
        analysis = {
            'feasible': feasible,
            'total_budget': total_budget,
            'per_person_budget': per_person_budget,
            'daily_budget': daily_budget,
            'min_required': min_required,
            'surplus_deficit': total_budget - min_required,
            'recommendation': ''
        }
        
        if feasible:
            if total_budget >= min_required * 2:
                analysis['recommendation'] = 'Excellent budget for a comfortable trip'
            elif total_budget >= min_required * 1.5:
                analysis['recommendation'] = 'Good budget with room for upgrades'
            else:
                analysis['recommendation'] = 'Budget is feasible but will be tight'
        else:
            shortfall = min_required - total_budget
            analysis['recommendation'] = f'Budget is ${shortfall:.2f} short of minimum requirements'
        
        logger.info(f"Feasibility: {feasible}, Daily budget: ${daily_budget:.2f}")
        return analysis


def create_budget_optimizer() -> BudgetOptimizer:
    """Factory function to create budget optimizer."""
    return BudgetOptimizer()
