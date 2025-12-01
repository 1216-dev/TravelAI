"""
Llama-based Orchestrator for intent parsing and constraint validation.
Acts as the main reasoning and planning agent.
"""

import json
from typing import Dict, List, Any, Optional
from datetime import datetime
import yaml
from backend.utils.logger import get_logger
from backend.utils.validators import ConstraintValidator

logger = get_logger(__name__)


class LlamaOrchestrator:
    """
    Orchestrator agent using Llama model for reasoning and planning.
    Parses user intent, validates constraints, and coordinates workflow.
    """
    
    def __init__(self, config_path: str = "backend/utils/config.yaml"):
        """Initialize orchestrator with configuration."""
        with open(config_path, 'r') as f:
            self.config = yaml.safe_load(f)
        
        self.model_config = self.config['models']['orchestrator']
        self.validator = ConstraintValidator(config_path)
        
        # For now, we'll use rule-based reasoning
        # In production, integrate with transformers library for Llama
        self.use_local_model = self.model_config.get('use_local', True)
        
        logger.info("Llama Orchestrator initialized")
    
    def parse_user_intent(self, request: Dict[str, Any]) -> Dict[str, Any]:
        """
        Parse and structure user intent from natural language query.
        
        Args:
            request: Raw user request
            
        Returns:
            Structured intent dictionary
        """
        logger.info("Parsing user intent")
        
        query = request.get('query', '')
        
        # Extract intent components
        intent = {
            'trip_type': self._classify_trip_type(query, request.get('preferences', {})),
            'primary_goal': self._extract_primary_goal(query),
            'flexibility': self._assess_flexibility(request),
            'priorities': self._rank_priorities(request),
            'constraints': self._extract_constraints(request),
            'parsed_at': datetime.now().isoformat()
        }
        
        logger.debug(f"Parsed intent: {intent['trip_type']} trip with goal: {intent['primary_goal']}")
        return intent
    
    def validate_request(self, request: Dict[str, Any]) -> Dict[str, Any]:
        """
        Validate travel request constraints.
        
        Args:
            request: Travel planning request
            
        Returns:
            Validation result with status and errors
        """
        logger.info("Validating travel request")
        
        is_valid, errors = self.validator.validate_all(request)
        
        validation_result = {
            'is_valid': is_valid,
            'errors': errors,
            'warnings': [],
            'suggestions': []
        }
        
        # Add suggestions for improvement
        if not is_valid:
            validation_result['suggestions'] = self._generate_suggestions(errors, request)
        
        # Add warnings for potential issues
        validation_result['warnings'] = self._check_warnings(request)
        
        if is_valid:
            logger.info("✅ Request validation passed")
        else:
            logger.warning(f"⚠️ Request validation failed: {len(errors)} errors")
        
        return validation_result
    
    def plan_workflow(self, request: Dict[str, Any], intent: Dict[str, Any]) -> Dict[str, Any]:
        """
        Plan the agent workflow based on request and intent.
        
        Args:
            request: Travel planning request
            intent: Parsed intent
            
        Returns:
            Workflow plan with agent sequence and parameters
        """
        logger.info("Planning agent workflow")
        
        workflow = {
            'stages': [],
            'parallelizable': [],
            'dependencies': {},
            'estimated_duration': 0
        }
        
        # Stage 1: Data Collection (parallelizable)
        workflow['stages'].append({
            'name': 'data_collection',
            'agents': ['api_manager'],
            'tasks': ['fetch_flights', 'fetch_hotels', 'fetch_weather'],
            'parallel': True,
            'estimated_duration': 3.0
        })
        
        # Stage 2: Personalization
        workflow['stages'].append({
            'name': 'personalization',
            'agents': ['gnn_agent'],
            'tasks': ['analyze_preferences', 'generate_recommendations'],
            'parallel': False,
            'depends_on': ['data_collection'],
            'estimated_duration': 2.0
        })
        
        # Stage 3: Budget Optimization
        workflow['stages'].append({
            'name': 'optimization',
            'agents': ['budget_optimizer'],
            'tasks': ['optimize_budget', 'generate_alternatives'],
            'parallel': False,
            'depends_on': ['personalization'],
            'estimated_duration': 1.5
        })
        
        # Stage 4: Itinerary Generation
        workflow['stages'].append({
            'name': 'itinerary_generation',
            'agents': ['itinerary_agent'],
            'tasks': ['consolidate_plan', 'generate_pdf'],
            'parallel': False,
            'depends_on': ['optimization'],
            'estimated_duration': 2.0
        })
        
        workflow['estimated_duration'] = sum(stage['estimated_duration'] for stage in workflow['stages'])
        
        logger.info(f"Workflow planned: {len(workflow['stages'])} stages, ~{workflow['estimated_duration']:.1f}s")
        return workflow
    
    def coordinate_agents(self, workflow: Dict[str, Any], context: Dict[str, Any]) -> Dict[str, Any]:
        """
        Coordinate agent execution according to workflow plan.
        
        Args:
            workflow: Planned workflow
            context: Execution context
            
        Returns:
            Coordination results
        """
        logger.info("Coordinating agent execution")
        
        results = {
            'completed_stages': [],
            'current_stage': None,
            'failed_stages': [],
            'agent_outputs': {},
            'status': 'in_progress'
        }
        
        # This is a planning function - actual execution happens in main.py
        results['execution_plan'] = workflow
        results['status'] = 'planned'
        
        return results
    
    def _classify_trip_type(self, query: str, preferences: Dict) -> str:
        """Classify the type of trip based on query and preferences."""
        query_lower = query.lower()
        pref_categories = preferences.get('categories', [])
        
        # Keywords for trip types
        trip_types = {
            'adventure': ['adventure', 'hiking', 'trekking', 'outdoor', 'sports'],
            'leisure': ['relax', 'beach', 'vacation', 'leisure', 'rest'],
            'business': ['business', 'conference', 'meeting', 'work'],
            'cultural': ['cultural', 'museum', 'historic', 'heritage'],
            'family': ['family', 'kids', 'children'],
            'romantic': ['romantic', 'honeymoon', 'couple'],
            'luxury': ['luxury', 'premium', 'upscale', 'high-end']
        }
        
        # Check preferences first
        for pref in pref_categories:
            if pref in trip_types:
                return pref
        
        # Check query keywords
        for trip_type, keywords in trip_types.items():
            if any(keyword in query_lower for keyword in keywords):
                return trip_type
        
        return 'leisure'  # Default
    
    def _extract_primary_goal(self, query: str) -> str:
        """Extract the primary goal from the query."""
        query_lower = query.lower()
        
        goals = {
            'explore': ['explore', 'discover', 'see', 'visit'],
            'relax': ['relax', 'unwind', 'rest', 'chill'],
            'experience': ['experience', 'try', 'taste', 'learn'],
            'celebrate': ['celebrate', 'anniversary', 'birthday', 'special'],
            'adventure': ['adventure', 'thrill', 'excitement']
        }
        
        for goal, keywords in goals.items():
            if any(keyword in query_lower for keyword in keywords):
                return goal
        
        return 'explore'  # Default
    
    def _assess_flexibility(self, request: Dict[str, Any]) -> Dict[str, bool]:
        """Assess flexibility in dates, budget, and preferences."""
        constraints = request.get('constraints', {})
        
        return {
            'dates': constraints.get('flexible_dates', False),
            'budget': request.get('budget', {}).get('flexible', False),
            'destination': False,  # Usually fixed
            'accommodation': True,  # Usually flexible
            'activities': True
        }
    
    def _rank_priorities(self, request: Dict[str, Any]) -> List[str]:
        """Rank user priorities."""
        preferences = request.get('preferences', {})
        comfort_level = preferences.get('comfort_level', 'standard')
        
        # Default priority rankings
        if comfort_level == 'budget':
            return ['price', 'value', 'location', 'comfort', 'amenities']
        elif comfort_level == 'luxury':
            return ['comfort', 'amenities', 'service', 'location', 'price']
        else:
            return ['value', 'location', 'comfort', 'price', 'amenities']
    
    def _extract_constraints(self, request: Dict[str, Any]) -> Dict[str, Any]:
        """Extract all constraints from request."""
        return {
            'dates': request.get('dates', {}),
            'budget': request.get('budget', {}),
            'group_size': request.get('group_size', 1),
            'special': request.get('constraints', {}),
            'preferences': request.get('preferences', {})
        }
    
    def _generate_suggestions(self, errors: List[str], request: Dict[str, Any]) -> List[str]:
        """Generate suggestions to fix validation errors."""
        suggestions = []
        
        for error in errors:
            if 'budget' in error.lower():
                suggestions.append("Consider adjusting your budget or travel dates for better options")
            elif 'date' in error.lower():
                suggestions.append("Try selecting dates further in the future")
            elif 'duration' in error.lower():
                suggestions.append("Adjust trip duration to meet requirements")
        
        return suggestions
    
    def _check_warnings(self, request: Dict[str, Any]) -> List[str]:
        """Check for potential issues that aren't errors."""
        warnings = []
        
        # Check budget vs destination
        budget = request.get('budget', {}).get('total', 0)
        duration = self.validator.get_trip_duration(
            request['dates']['start'],
            request['dates']['end']
        )
        
        daily_budget = budget / duration if duration > 0 else 0
        
        if daily_budget < 100:
            warnings.append("Low daily budget may limit accommodation and activity options")
        
        # Check advance booking
        start_date = datetime.strptime(request['dates']['start'], "%Y-%m-%d")
        days_advance = (start_date - datetime.now()).days
        
        if days_advance < 14:
            warnings.append("Booking with less than 2 weeks advance may result in higher prices")
        
        return warnings


def create_orchestrator() -> LlamaOrchestrator:
    """Factory function to create orchestrator instance."""
    return LlamaOrchestrator()
