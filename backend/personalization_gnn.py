"""
GNN-based Personalization Agent for user preference reasoning.
Uses Graph Neural Networks with Neo4j backend for persistent graph storage and real ML.
"""

import random
import time
import torch
import torch.nn.functional as F
from typing import Dict, List, Any, Tuple
import yaml
import numpy as np
from neo4j import GraphDatabase
from torch_geometric.data import Data
from torch_geometric.nn import GCNConv, GATConv
from backend.utils.logger import get_logger

logger = get_logger(__name__)


class RealGNNModel(torch.nn.Module):
    """Real Graph Neural Network model using PyTorch Geometric"""
    
    def __init__(self, input_dim=64, hidden_dim=128, output_dim=64, num_layers=3, heads=8):
        super().__init__()
        self.num_layers = num_layers
        
        # Use GAT (Graph Attention Network) for better performance
        self.convs = torch.nn.ModuleList()
        self.convs.append(GATConv(input_dim, hidden_dim // heads, heads=heads, dropout=0.2))
        
        for _ in range(num_layers - 2):
            self.convs.append(GATConv(hidden_dim, hidden_dim // heads, heads=heads, dropout=0.2))
        
        self.convs.append(GATConv(hidden_dim, output_dim, heads=1, dropout=0.2))
        
        # Final projection layer
        self.projection = torch.nn.Linear(output_dim, output_dim)
        
    def forward(self, x, edge_index, batch=None):
        # Apply GNN layers
        for i, conv in enumerate(self.convs):
            x = conv(x, edge_index)
            if i < len(self.convs) - 1:  # Don't apply activation on last layer
                x = F.elu(x)
                x = F.dropout(x, p=0.2, training=self.training)
        
        # Global mean pooling if batch is provided
        if batch is not None:
            x = torch_geometric.nn.global_mean_pool(x, batch)
        
        # Final projection
        x = self.projection(x)
        return F.normalize(x, p=2, dim=-1)


class Neo4jPersonalizationGNN:
    """
    Enhanced Graph Neural Network agent with Neo4j backend.
    Provides persistent graph storage and real GNN models for personalized recommendations.
    """
    
    def __init__(self, config_path: str = "backend/utils/config.yaml"):
        """Initialize Neo4j-enhanced GNN personalization agent."""
        with open(config_path, 'r') as f:
            self.config = yaml.safe_load(f)
        
        self.gnn_config = self.config['models']['gnn']
        self.pref_config = self.config['personalization']
        self.neo4j_config = self.config.get('neo4j', {
            'uri': 'bolt://localhost:7687',
            'user': 'neo4j',
            'password': 'password'
        })
        
        # Initialize Neo4j connection
        self.driver = GraphDatabase.driver(
            self.neo4j_config['uri'], 
            auth=(self.neo4j_config['user'], self.neo4j_config['password'])
        )
        
        # Initialize PyTorch device and model
        self.device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
        self.model = self._create_gnn_model()
        self.model_loaded = False
        
        # Cache for embeddings and graph data
        self.user_embeddings = {}
        self.node_id_map = {}
        self.edge_index_cache = {}
        
        logger.info("Neo4j-enhanced GNN Personalization Agent initialized")
        self._create_neo4j_schema()
    
    def _create_gnn_model(self):
        """Create the actual GNN model"""
        return RealGNNModel(
            input_dim=self.pref_config['user_embedding_dim'],
            hidden_dim=self.gnn_config['hidden_dim'],
            output_dim=self.pref_config['user_embedding_dim'],
            num_layers=self.gnn_config['num_layers']
        ).to(self.device)
    
    def _create_neo4j_schema(self):
        """Create Neo4j graph schema for travel recommendations"""
        try:
            with self.driver.session() as session:
                # Create constraints for unique nodes
                session.run("CREATE CONSTRAINT user_id IF NOT EXISTS ON (u:User) ASSERT u.id IS UNIQUE")
                session.run("CREATE CONSTRAINT destination_name IF NOT EXISTS ON (d:Destination) ASSERT d.name IS UNIQUE")
                session.run("CREATE CONSTRAINT activity_id IF NOT EXISTS ON (a:Activity) ASSERT a.id IS UNIQUE")
                session.run("CREATE CONSTRAINT preference_category IF NOT EXISTS ON (p:Preference) ASSERT p.category IS UNIQUE")
                
                logger.info("Neo4j schema created successfully")
        except Exception as e:
            logger.error(f"Failed to create Neo4j schema: {e}")

    def _normalize_preferences(self, preferences: Any) -> List[str]:
        """Normalize preferences into a simple list of strings for scoring/logging."""
        if isinstance(preferences, dict):
            if 'categories' in preferences and isinstance(preferences['categories'], list):
                return preferences['categories']
            return [k for k, v in preferences.items() if v]
        if isinstance(preferences, list):
            return preferences
        return []
    
    def build_enhanced_preference_graph(
        self,
        user_id: str,
        preferences: Dict[str, Any],
        history: Dict[str, Any]
    ) -> Dict[str, Any]:
        """
        Build an enhanced preference graph in Neo4j.
        
        Args:
            user_id: User identifier
            preferences: Current preferences
            history: User travel history
            
        Returns:
            Graph representation stored in Neo4j
        """
        logger.info(f"Building enhanced preference graph for user {user_id}")
        
        try:
            with self.driver.session() as session:
                # Create user node with detailed properties
                session.run("""
                    MERGE (u:User {id: $user_id})
                    SET u.budget_preference = $budget,
                        u.group_size = $group_size,
                        u.comfort_level = $comfort_level,
                        u.last_updated = datetime()
                """, 
                user_id=user_id,
                budget=preferences.get('budget', 1000),
                group_size=preferences.get('group_size', 1),
                comfort_level=preferences.get('comfort_level', 'standard'))
                
                # Create preference nodes and relationships
                pref_categories = preferences.get('categories', [])
                for category in pref_categories:
                    session.run("""
                        MERGE (p:Preference {category: $category})
                        MERGE (u:User {id: $user_id})
                        MERGE (u)-[:LIKES {strength: 1.0, created_at: datetime()}]->(p)
                    """, category=category, user_id=user_id)
                
                # Create destination nodes from history
                prev_trips = history.get('previous_trips', [])
                for i, trip in enumerate(prev_trips):
                    dest_name = trip.get('destination')
                    satisfaction = trip.get('satisfaction', 0)
                    
                    # Create destination node
                    session.run("""
                        MERGE (d:Destination {name: $name})
                        SET d.satisfaction_avg = CASE 
                            WHEN d.satisfaction_avg IS NULL THEN $satisfaction 
                            ELSE (d.satisfaction_avg + $satisfaction) / 2.0 
                        END,
                        d.visit_count = CASE 
                            WHEN d.visit_count IS NULL THEN 1 
                            ELSE d.visit_count + 1 
                        END
                    """, name=dest_name, satisfaction=satisfaction)
                    
                    # Create user-destination relationship
                    session.run("""
                        MATCH (u:User {id: $user_id})
                        MATCH (d:Destination {name: $dest_name})
                        MERGE (u)-[v:VISITED {satisfaction: $satisfaction, trip_index: $trip_index}]->(d)
                    """, user_id=user_id, dest_name=dest_name, satisfaction=satisfaction, trip_index=i)
                    
                    # Connect destination to preferences
                    trip_preferences = trip.get('preferences', [])
                    for pref in trip_preferences:
                        session.run("""
                            MATCH (d:Destination {name: $dest_name})
                            MATCH (p:Preference {category: $pref})
                            MERGE (d)-[:HAS_PREFERENCE {weight: 0.8}]->(p)
                        """, dest_name=dest_name, pref=pref)
                
                logger.info(f"Enhanced graph built for user {user_id}")
                return {"status": "success", "nodes_created": len(pref_categories) + len(prev_trips) + 1}
                
        except Exception as e:
            logger.error(f"Failed to build enhanced preference graph: {e}")
            return {"status": "error", "message": str(e)}
    
    def generate_user_embedding(
        self,
        user_id: str,
        preference_graph: Dict[str, Any] = None
    ) -> List[float]:
        """
        Generate user embedding using trained GNN model.
        
        Args:
            user_id: User identifier
            preference_graph: User's preference graph (optional)
            
        Returns:
            User embedding vector
        """
        logger.info(f"Generating user embedding for {user_id}")
        
        # Check cache first
        if user_id in self.user_embeddings:
            logger.debug("Using cached embedding")
            return self.user_embeddings[user_id]
        
        try:
            # Extract user subgraph from Neo4j
            graph_data = self._extract_user_graph_data(user_id)
            
            if graph_data is None or graph_data['num_nodes'] == 0:
                logger.warning(f"No graph data found for user {user_id}")
                return self._create_fallback_embedding()
            
            # Create PyTorch Geometric data object
            x, edge_index = self._prepare_tensor_data(graph_data)
            
            # Generate embedding using GNN model
            self.model.eval()
            with torch.no_grad():
                embedding_tensor = self.model(x, edge_index)
            
            # Convert to Python list
            embedding = embedding_tensor.cpu().numpy().tolist()
            
            # Cache embedding
            self.user_embeddings[user_id] = embedding
            
            logger.info(f"Generated {len(embedding)}-dimensional embedding")
            return embedding
            
        except Exception as e:
            logger.error(f"Failed to generate embedding: {e}")
            return self._create_fallback_embedding()
    
    def _extract_user_graph_data(self, user_id: str) -> Dict[str, Any]:
        """Extract user's graph data from Neo4j"""
        try:
            with self.driver.session() as session:
                # Get all nodes and relationships for the user
                result = session.run("""
                    MATCH (u:User {id: $user_id})
                    OPTIONAL MATCH (u)-[r1:LIKES]->(p:Preference)
                    OPTIONAL MATCH (u)-[r2:VISITED]->(d:Destination)
                    OPTIONAL MATCH (d)-[r3:HAS_PREFERENCE]->(p2:Preference)
                    RETURN 
                        collect(DISTINCT u) as users,
                        collect(DISTINCT p) as preferences,
                        collect(DISTINCT d) as destinations,
                        collect(DISTINCT r1) + collect(DISTINCT r2) + collect(DISTINCT r3) as relationships
                """, user_id=user_id)
                
                record = result.single()
                
                if not record:
                    return None
                
                users = record['users']
                preferences = record['preferences'] 
                destinations = record['destinations']
                relationships = record['relationships']
                
                # Create node feature matrices
                all_nodes = users + preferences + destinations
                num_nodes = len(all_nodes)
                
                if num_nodes == 0:
                    return None
                
                # Create node features (simplified)
                node_features = []
                for node in all_nodes:
                    if node['id'] == user_id:  # User node
                        features = [1.0] * 64  # User gets all-1 features
                    elif 'category' in node:  # Preference node
                        features = self._encode_preference(node['category'])
                    else:  # Destination node
                        features = self._encode_destination(node)
                    
                    node_features.append(features)
                
                # Create edge index
                edge_index = []
                node_id_to_idx = {node['id']: idx for idx, node in enumerate(all_nodes)}
                
                for rel in relationships:
                    if rel.start_node['id'] in node_id_to_idx and rel.end_node['id'] in node_id_to_idx:
                        edge_index.extend([
                            [node_id_to_idx[rel.start_node['id']]],
                            [node_id_to_idx[rel.end_node['id']]]
                        ])
                
                if not edge_index:
                    edge_index = [[], []]
                
                return {
                    'num_nodes': num_nodes,
                    'node_features': node_features,
                    'edge_index': edge_index,
                    'all_nodes': all_nodes
                }
                
        except Exception as e:
            logger.error(f"Failed to extract graph data: {e}")
            return None
    
    def _encode_preference(self, category: str) -> List[float]:
        """Encode preference category into feature vector"""
        pref_categories = self.pref_config['preference_categories']
        encoding = [0.0] * len(pref_categories)
        
        if category in pref_categories:
            idx = pref_categories.index(category)
            encoding[idx] = 1.0
        
        # Pad to 64 dimensions
        while len(encoding) < 64:
            encoding.append(0.0)
        
        return encoding[:64]
    
    def _encode_destination(self, destination_node: Dict) -> List[float]:
        """Encode destination into feature vector"""
        features = [0.0] * 64
        
        # Use satisfaction score
        satisfaction = destination_node.get('satisfaction_avg', 0)
        features[0] = satisfaction / 5.0
        
        # Use visit count (normalized)
        visit_count = destination_node.get('visit_count', 0)
        features[1] = min(visit_count / 10.0, 1.0)  # Cap at 10 visits
        
        return features
    
    def _prepare_tensor_data(self, graph_data: Dict) -> Tuple[torch.Tensor, torch.Tensor]:
        """Prepare tensor data for GNN model"""
        x = torch.tensor(graph_data['node_features'], dtype=torch.float).to(self.device)
        edge_index = torch.tensor(graph_data['edge_index'], dtype=torch.long).to(self.device)
        return x, edge_index
    
    def _create_fallback_embedding(self) -> List[float]:
        """Create fallback embedding when graph data is unavailable"""
        embedding_dim = self.pref_config['user_embedding_dim']
        embedding = [random.random() for _ in range(embedding_dim)]
        
        # Normalize
        norm = sum(x**2 for x in embedding) ** 0.5
        embedding = [x / norm for x in embedding]
        
        return embedding
    
    def rank_options(
        self,
        user_embedding: List[float],
        options: List[Dict[str, Any]],
        option_type: str = 'hotel'
    ) -> List[Dict[str, Any]]:
        """
        Rank options using GNN-generated user embedding.
        
        Args:
            user_embedding: User preference embedding from GNN
            options: List of options (hotels, flights, activities)
            option_type: Type of options being ranked
            
        Returns:
            Ranked list of options with personalization scores
        """
        logger.info(f"Ranking {len(options)} {option_type} options using GNN embeddings")
        
        scored_options = []
        
        for option in options:
            if not isinstance(option, dict):
                logger.warning(f"Skipping non-dict option: {type(option)}")
                continue
                
            # Calculate personalization score using embedding similarity
            score = self._calculate_embedding_similarity_score(
                user_embedding, option, option_type
            )
            
            option_with_score = option.copy()
            option_with_score['personalization_score'] = score
            scored_options.append(option_with_score)
        
        # Sort by personalization score
        if scored_options:
            scored_options.sort(key=lambda x: x['personalization_score'], reverse=True)
            logger.info(f"Top option score: {scored_options[0]['personalization_score']:.3f}")
        
        return scored_options
    
    def _calculate_embedding_similarity_score(
        self,
        user_embedding: List[float],
        option: Dict[str, Any],
        option_type: str
    ) -> float:
        """Calculate personalization score using embedding similarity"""
        
        # Create option embedding based on its features
        option_embedding = self._create_option_embedding(option, option_type)
        
        # Calculate cosine similarity
        similarity = self._cosine_similarity(user_embedding, option_embedding)
        
        # Combine with traditional scoring
        base_score = self._calculate_base_score(option, option_type)
        
        # Weighted combination
        final_score = 0.6 * similarity + 0.4 * base_score
        
        return min(final_score, 1.0)
    
    def _create_option_embedding(self, option: Dict, option_type: str) -> List[float]:
        """Create embedding for an option based on its features"""
        embedding_dim = self.pref_config['user_embedding_dim']
        embedding = [0.0] * embedding_dim
        
        if option_type == 'hotel':
            # Use rating and amenities
            rating = option.get('rating', 3.5) / 5.0
            embedding[0] = rating
            
            amenities = option.get('amenities', [])
            embedding[1] = min(len(amenities) / 10.0, 1.0)  # Normalize amenities count
            
        elif option_type == 'flight':
            # Use duration and stops
            duration = option.get('duration', 0)
            stops = option.get('stops', 0)
            
            # Shorter duration and fewer stops are better
            embedding[0] = max(0, 1.0 - (duration / 20.0))  # Normalize to 20h max
            embedding[1] = max(0, 1.0 - (stops * 0.3))  # Fewer stops better
            
        elif option_type == 'activity':
            # Use rating and price level
            rating = option.get('rating', 4.0) / 5.0
            price_level = option.get('price_level', 2)
            
            embedding[0] = rating
            embedding[1] = max(0, 1.0 - ((price_level - 1) * 0.25))  # Lower price better
        
        return embedding
    
    def _cosine_similarity(self, vec1: List[float], vec2: List[float]) -> float:
        """Calculate cosine similarity between two vectors"""
        if len(vec1) != len(vec2):
            return 0.0
            
        dot_product = sum(a * b for a, b in zip(vec1, vec2))
        norm1 = sum(a * a for a in vec1) ** 0.5
        norm2 = sum(b * b for b in vec2) ** 0.5
        
        if norm1 == 0 or norm2 == 0:
            return 0.0
            
        return dot_product / (norm1 * norm2)
    
    def _calculate_base_score(self, option: Dict, option_type: str) -> float:
        """Calculate base score for an option"""
        if option_type == 'hotel':
            rating = option.get('rating', 3.5)
            return rating / 5.0
            
        elif option_type == 'flight':
            stops = option.get('stops', 0)
            return max(0, 1.0 - (stops * 0.2))  # Fewer stops better
            
        elif option_type == 'activity':
            rating = option.get('rating', 4.0)
            return rating / 5.0
            
        return 0.5
    
    def recommend_activities(
        self,
        user_profile: Dict[str, Any],
        destination: str,
        available_activities: List[Dict] = None
    ) -> List[Dict[str, Any]]:
        """
        Recommend activities using GNN-based personalization.
        
        Args:
            user_profile: User preferences and profile
            destination: Destination city
            available_activities: Pre-fetched activities from API
            
        Returns:
            List of recommended activities
        """
        logger.info(f"Generating activity recommendations for {destination}")
        
        # Get user embedding
        user_id = user_profile.get('user_id', 'anonymous')
        user_embedding = self.generate_user_embedding(user_id)
        preferences = self._normalize_preferences(user_profile.get('preferences', []))
        
        # Use provided activities or generate enhanced mock ones
        if available_activities:
            activities = available_activities
        else:
            activities = self._generate_enhanced_mock_activities(user_profile, destination)
        
        # Score activities using GNN embeddings
        for activity in activities:
            activity['personalization_score'] = self._calculate_activity_embedding_score(
                activity, user_embedding, preferences
            )
        
        # Sort by score and return top activities
        activities.sort(key=lambda x: x.get('personalization_score', 0), reverse=True)
        
        top_activities = activities[:10]
        # Persist a small snapshot for observability
        self._record_personalization_snapshot(
            user_id=user_id,
            destination=destination,
            preferences=preferences,
            user_embedding=user_embedding,
            top_activities=top_activities[:5]
        )
        
        return top_activities
    
    def _calculate_activity_embedding_score(
        self, 
        activity: Dict, 
        user_embedding: List[float],
        preferences: List[str]
    ) -> float:
        """Calculate personalization score for activity using embeddings"""
        
        # Create activity embedding
        activity_embedding = self._create_option_embedding(activity, 'activity')
        
        # Calculate similarity
        similarity = self._cosine_similarity(user_embedding, activity_embedding)
        
        # Boost score if activity category matches user preferences
        activity_category = activity.get('category', '').lower()
        category_match = any(pref.lower() in activity_category for pref in preferences)
        
        category_bonus = 0.2 if category_match else 0.0
        
        # Base score from rating
        base_rating = activity.get('rating', 4.0) / 5.0
        
        # Combine scores
        final_score = 0.5 * similarity + 0.3 * base_rating + category_bonus
        
        return min(final_score, 1.0)

    def _record_personalization_snapshot(
        self,
        user_id: str,
        destination: str,
        preferences: List[str],
        user_embedding: List[float],
        top_activities: List[Dict[str, Any]]
    ) -> None:
        """
        Persist a quick snapshot of what the GNN considered so we can inspect personalization in Neo4j.
        """
        if not top_activities:
            return
        
        try:
            run_id = f"{user_id}-{destination}-{int(time.time())}"
            preview_embedding = user_embedding[:8]  # keep a short preview to avoid huge properties
            
            with self.driver.session() as session:
                session.run(
                    """
                    MERGE (u:User {id: $user_id})
                    SET u.last_seen = datetime(),
                        u.preference_tags = $preferences,
                        u.embedding_preview = $embedding
                    """,
                    user_id=user_id,
                    preferences=preferences,
                    embedding=preview_embedding
                )
                
                session.run(
                    """
                    MERGE (ctx:PersonalizationRun {id: $run_id})
                    SET ctx.destination = $destination,
                        ctx.seen_at = datetime(),
                        ctx.preference_tags = $preferences
                    """,
                    run_id=run_id,
                    destination=destination,
                    preferences=preferences
                )
                
                session.run(
                    """
                    MATCH (u:User {id: $user_id})
                    MATCH (ctx:PersonalizationRun {id: $run_id})
                    MERGE (u)-[:PERSONALIZED_AT]->(ctx)
                    """,
                    user_id=user_id,
                    run_id=run_id
                )
                
                for rank, activity in enumerate(top_activities, 1):
                    activity_id = (
                        activity.get('id')
                        or activity.get('activity_id')
                        or f"{destination.lower()}_{activity.get('name', 'activity').lower().replace(' ', '_')}"
                    )
                    score = round(activity.get('personalization_score', 0), 3)
                    reason = activity.get('description', 'Matched to preferences')
                    
                    session.run(
                        """
                        MERGE (a:Activity {id: $activity_id})
                        SET a.name = $name,
                            a.category = $category,
                            a.destination = $destination,
                            a.rating = $rating,
                            a.last_seen = datetime()
                        """,
                        activity_id=activity_id,
                        name=activity.get('name'),
                        category=activity.get('category'),
                        destination=destination,
                        rating=activity.get('rating', 0)
                    )
                    
                    session.run(
                        """
                        MATCH (ctx:PersonalizationRun {id: $run_id})
                        MATCH (a:Activity {id: $activity_id})
                        MERGE (ctx)-[r:RECOMMENDED {rank: $rank}]->(a)
                        SET r.score = $score,
                            r.reason = $reason,
                            r.seen_at = datetime()
                        """,
                        run_id=run_id,
                        activity_id=activity_id,
                        rank=rank,
                        score=score,
                        reason=reason
                    )
        except Exception as e:
            logger.warning(f"Failed to log personalization snapshot: {e}")
    
    def _generate_enhanced_mock_activities(self, user_profile: Dict, destination: str) -> List[Dict]:
        """Generate enhanced mock activities with better features"""
        activities = []
        
        # Activity templates based on destination and preferences
        activity_templates = {
            'adventure': [
                {'name': 'Hiking Adventure', 'category': 'adventure', 'rating': 4.5, 'price_level': 2},
                {'name': 'Zip-lining Experience', 'category': 'adventure', 'rating': 4.8, 'price_level': 3},
                {'name': 'Kayaking Tour', 'category': 'adventure', 'rating': 4.3, 'price_level': 2},
            ],
            'cultural': [
                {'name': 'Museum Visit', 'category': 'cultural', 'rating': 4.2, 'price_level': 1},
                {'name': 'Historical Walking Tour', 'category': 'cultural', 'rating': 4.4, 'price_level': 1},
                {'name': 'Art Gallery Exploration', 'category': 'cultural', 'rating': 4.1, 'price_level': 1},
            ],
            'culinary': [
                {'name': 'Food Tour', 'category': 'culinary', 'rating': 4.6, 'price_level': 2},
                {'name': 'Cooking Class', 'category': 'culinary', 'rating': 4.7, 'price_level': 3},
                {'name': 'Wine Tasting', 'category': 'culinary', 'rating': 4.5, 'price_level': 3},
            ],
            'nature': [
                {'name': 'Nature Walk', 'category': 'nature', 'rating': 4.0, 'price_level': 1},
                {'name': 'Botanical Garden Visit', 'category': 'nature', 'rating': 4.2, 'price_level': 1},
                {'name': 'Scenic Viewpoint', 'category': 'nature', 'rating': 4.3, 'price_level': 1},
            ],
            'relaxation': [
                {'name': 'Spa Day', 'category': 'relaxation', 'rating': 4.4, 'price_level': 3},
                {'name': 'Yoga Class', 'category': 'relaxation', 'rating': 4.2, 'price_level': 2},
                {'name': 'Beach Relaxation', 'category': 'relaxation', 'rating': 4.1, 'price_level': 1},
            ]
        }
        
        # Generate activities based on user preferences
        user_preferences = user_profile.get('preferences', [])
        
        for pref in user_preferences:
            if pref in activity_templates:
                for template in activity_templates[pref][:2]:  # Take first 2 activities per preference
                    activity = template.copy()
                    activity['id'] = f"{destination.lower()}_{activity['name'].lower().replace(' ', '_')}"
                    activity['destination'] = destination
                    activities.append(activity)
        
        # Add some general activities if no preferences match
        if not activities:
            general_activities = [
                {'name': 'City Tour', 'category': 'cultural', 'rating': 4.0, 'price_level': 2},
                {'name': 'Local Market Visit', 'category': 'cultural', 'rating': 4.2, 'price_level': 1},
                {'name': 'Scenic Walk', 'category': 'nature', 'rating': 4.1, 'price_level': 1},
            ]
            
            for activity in general_activities:
                activity['id'] = f"{destination.lower()}_{activity['name'].lower().replace(' ', '_')}"
                activity['destination'] = destination
                activities.append(activity)
        
        return activities[:8]  # Return up to 8 activities
    
    def explain_recommendation(
        self,
        option: Dict[str, Any],
        user_id: str
    ) -> str:
        """
        Generate explanation for why an option was recommended using GNN insights.
        
        Args:
            option: Recommended option
            user_id: User identifier
            
        Returns:
            Human-readable explanation
        """
        score = option.get('personalization_score', 0)
        option_type = option.get('type', 'option')
        
        if score > 0.85:
            return f"Perfect {option_type} match based on your preferences and travel history"
        elif score > 0.70:
            return f"Great {option_type} fit - aligns well with your past choices"
        elif score > 0.55:
            return f"Good {option_type} option that matches your interests"
        else:
            return f"Recommended {option_type} based on popularity and general appeal"
    
    def cleanup(self):
        """Clean up Neo4j connection"""
        if self.driver:
            self.driver.close()
            logger.info("Neo4j connection closed")


# Backward compatibility factory function
def create_personalization_agent() -> Neo4jPersonalizationGNN:
    """Factory function to create enhanced personalization agent."""
    return Neo4jPersonalizationGNN()
