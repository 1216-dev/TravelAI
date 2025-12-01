"""
Constraint validators for travel planning requests.
Validates timing, budget, and feasibility constraints.
"""

from datetime import datetime, timedelta
from typing import Dict, List, Tuple, Optional
import yaml
import os


class ConstraintValidator:
    """Validates travel planning constraints."""
    
    def __init__(self, config_path: str = "backend/utils/config.yaml"):
        """Initialize validator with configuration."""
        with open(config_path, 'r') as f:
            self.config = yaml.safe_load(f)
        self.constraints = self.config['constraints']
    
    def validate_dates(self, start_date: str, end_date: str) -> Tuple[bool, Optional[str]]:
        """
        Validate trip dates.
        
        Args:
            start_date: Start date in YYYY-MM-DD format
            end_date: End date in YYYY-MM-DD format
            
        Returns:
            Tuple of (is_valid, error_message)
        """
        try:
            start = datetime.strptime(start_date, "%Y-%m-%d")
            end = datetime.strptime(end_date, "%Y-%m-%d")
            now = datetime.now()
            
            # Check if dates are in the past
            if start < now:
                return False, "Start date cannot be in the past"
            
            # Check if end is after start
            if end <= start:
                return False, "End date must be after start date"
            
            # Check trip duration
            duration = (end - start).days
            min_days = self.constraints['min_trip_duration_days']
            max_days = self.constraints['max_trip_duration_days']
            
            if duration < min_days:
                return False, f"Trip must be at least {min_days} days"
            
            if duration > max_days:
                return False, f"Trip cannot exceed {max_days} days"
            
            # Check advance booking requirement
            advance_days = (start - now).days
            min_advance = self.constraints['advance_booking_days']
            
            if advance_days < min_advance:
                return False, f"Trips must be booked at least {min_advance} days in advance"
            
            return True, None
            
        except ValueError as e:
            return False, f"Invalid date format: {str(e)}"
    
    def validate_budget(self, budget: float, currency: str = "USD") -> Tuple[bool, Optional[str]]:
        """
        Validate budget constraints.
        
        Args:
            budget: Total budget amount
            currency: Currency code
            
        Returns:
            Tuple of (is_valid, error_message)
        """
        min_budget = self.constraints['min_budget']
        max_budget = self.constraints['max_budget']
        
        if budget < min_budget:
            return False, f"Budget must be at least {min_budget} {currency}"
        
        if budget > max_budget:
            return False, f"Budget cannot exceed {max_budget} {currency}"
        
        return True, None
    
    def validate_group_size(self, group_size: int) -> Tuple[bool, Optional[str]]:
        """Validate group size."""
        if group_size < 1:
            return False, "Group size must be at least 1"
        
        if group_size > 20:
            return False, "Group size cannot exceed 20 people"
        
        return True, None
    
    def validate_preferences(self, preferences: List[str]) -> Tuple[bool, Optional[str]]:
        """Validate preference categories."""
        valid_preferences = self.config['personalization']['preference_categories']
        
        invalid = [p for p in preferences if p not in valid_preferences]
        
        if invalid:
            return False, f"Invalid preferences: {', '.join(invalid)}"
        
        return True, None
    
    def validate_all(self, request: Dict) -> Tuple[bool, List[str]]:
        """
        Validate all constraints in a request.
        
        Args:
            request: Full travel planning request dictionary
            
        Returns:
            Tuple of (is_valid, list_of_errors)
        """
        errors = []
        
        # Validate dates
        if 'dates' in request:
            valid, error = self.validate_dates(
                request['dates']['start'],
                request['dates']['end']
            )
            if not valid:
                errors.append(error)
        
        # Validate budget
        if 'budget' in request:
            valid, error = self.validate_budget(
                request['budget']['total'],
                request['budget'].get('currency', 'USD')
            )
            if not valid:
                errors.append(error)
        
        # Validate group size
        if 'group_size' in request:
            valid, error = self.validate_group_size(request['group_size'])
            if not valid:
                errors.append(error)
        
        # Validate preferences
        if 'preferences' in request and 'categories' in request['preferences']:
            valid, error = self.validate_preferences(
                request['preferences']['categories']
            )
            if not valid:
                errors.append(error)
        
        return len(errors) == 0, errors
    
    def get_trip_duration(self, start_date: str, end_date: str) -> int:
        """Calculate trip duration in days."""
        start = datetime.strptime(start_date, "%Y-%m-%d")
        end = datetime.strptime(end_date, "%Y-%m-%d")
        return (end - start).days


def validate_request(request: Dict) -> Tuple[bool, List[str]]:
    """
    Convenience function to validate a travel request.
    
    Args:
        request: Travel planning request dictionary
        
    Returns:
        Tuple of (is_valid, list_of_errors)
    """
    validator = ConstraintValidator()
    return validator.validate_all(request)
