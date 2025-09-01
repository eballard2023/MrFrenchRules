"""
MongoDB Database Connection and Models
"""

import os
import asyncio
from datetime import datetime, timezone
from typing import List, Dict, Any, Optional
from motor.motor_asyncio import AsyncIOMotorClient, AsyncIOMotorDatabase, AsyncIOMotorCollection
from pydantic import BaseModel, Field, ConfigDict
from bson import ObjectId
import logging

logger = logging.getLogger(__name__)

class InterviewModel(BaseModel):
    """Model for interview sessions"""
    id: Optional[ObjectId] = Field(default=None, alias="_id")
    session_id: str
    started_at: datetime
    completed_at: Optional[datetime] = None
    conversation_history: List[Dict[str, str]] = Field(default_factory=list)
    questions_asked: int = 0
    is_complete: bool = False
    status: str = "in_progress"  # in_progress, completed, cancelled
    
    model_config = ConfigDict(
        populate_by_name=True,
        arbitrary_types_allowed=True,
        json_encoders={ObjectId: str},
    )

class ExtractedRuleModel(BaseModel):
    """Model for extracted behavioral rules"""
    id: Optional[ObjectId] = Field(default=None, alias="_id")
    interview_session_id: str
    rule_id: str
    trigger: Dict[str, Any]
    action: Dict[str, Any]
    priority: str
    category: str
    extracted_at: datetime
    
    model_config = ConfigDict(
        populate_by_name=True,
        arbitrary_types_allowed=True,
        json_encoders={ObjectId: str},
    )

class RulesCollectionModel(BaseModel):
    """Model for a collection of rules from one interview"""
    id: Optional[ObjectId] = Field(default=None, alias="_id")
    interview_session_id: str
    rules: List[Dict[str, Any]]
    total_rules: int
    extracted_at: datetime
    filename: str
    
    model_config = ConfigDict(
        populate_by_name=True,
        arbitrary_types_allowed=True,
        json_encoders={ObjectId: str},
    )

class DatabaseManager:
    """MongoDB Database Manager"""
    
    def __init__(self):
        self.client: Optional[AsyncIOMotorClient] = None
        self.database: Optional[AsyncIOMotorDatabase] = None
        self.interviews_collection: Optional[AsyncIOMotorCollection] = None
        self.rules_collection: Optional[AsyncIOMotorCollection] = None
        self.rules_collections_collection: Optional[AsyncIOMotorCollection] = None
        
    async def connect(self):
        """Connect to MongoDB"""
        try:
            # Get connection settings from environment
            connection_string = os.getenv("MONGODB_CONNECTION_STRING", "mongodb://localhost:27017")
            database_name = os.getenv("MONGODB_DATABASE_NAME", "ai_coach_interviews")
            interviews_collection_name = os.getenv("MONGODB_INTERVIEWS_COLLECTION", "interviews")
            rules_collection_name = os.getenv("MONGODB_RULES_COLLECTION", "extracted_rules")
            
            # Create client and connect
            self.client = AsyncIOMotorClient(
                connection_string,
                serverSelectionTimeoutMS=8000,
                connectTimeoutMS=8000,
                socketTimeoutMS=8000,
            )
            
            # Test the connection with more reasonable timeout for Atlas
            try:
                await asyncio.wait_for(self.client.admin.command('ping'), timeout=10.0)
                logger.info("Successfully connected to MongoDB")
            except asyncio.TimeoutError:
                logger.warning("MongoDB ping timed out; continuing without DB")
                return False
            except Exception as e:
                logger.error(f"MongoDB connection failed: {e}")
                return False
            
            # Get database and collections
            self.database = self.client[database_name]
            self.interviews_collection = self.database[interviews_collection_name]
            self.rules_collection = self.database[rules_collection_name]
            self.rules_collections_collection = self.database["rules_collections"]
            
            # Create indexes for better performance
            await self.create_indexes()
            
            return True
            
        except Exception as e:
            logger.error(f"Failed to connect to MongoDB: {e}")
            return False
    
    async def disconnect(self):
        """Disconnect from MongoDB"""
        if self.client:
            self.client.close()
            logger.info("Disconnected from MongoDB")
    
    async def create_indexes(self):
        """Create database indexes for better performance"""
        try:
            # Indexes for interviews collection
            await self.interviews_collection.create_index("session_id", unique=True)
            await self.interviews_collection.create_index("started_at")
            await self.interviews_collection.create_index("status")
            
            # Indexes for rules collection
            await self.rules_collection.create_index("interview_session_id")
            await self.rules_collection.create_index("rule_id")
            await self.rules_collection.create_index("category")
            await self.rules_collection.create_index("priority")
            
            # Indexes for rules collections - handle existing unique index
            try:
                await self.rules_collections_collection.create_index("interview_session_id")
            except Exception as index_error:
                if "IndexKeySpecsConflict" in str(index_error):
                    logger.info("Rules collection index already exists, skipping creation")
                else:
                    raise index_error
                    
            await self.rules_collections_collection.create_index("extracted_at")
            
            logger.info("Database indexes created successfully")
            
        except Exception as e:
            logger.error(f"Failed to create indexes: {e}")
    
    async def save_interview(self, interview_data: InterviewModel) -> str:
        """Save interview session to database"""
        try:
            interview_dict = interview_data.model_dump(by_alias=True, exclude={"id"})
            result = await self.interviews_collection.insert_one(interview_dict)
            logger.info(f"Interview saved with ID: {result.inserted_id}")
            return str(result.inserted_id)
        except Exception as e:
            logger.error(f"Failed to save interview: {e}")
            raise
    
    async def update_interview(self, session_id: str, update_data: Dict[str, Any]) -> bool:
        """Update interview session (upsert if missing)."""
        try:
            update_doc: Dict[str, Any] = {
                "$set": update_data,
                "$setOnInsert": {
                    "session_id": session_id,
                    "started_at": datetime.now(timezone.utc),
                    "status": update_data.get("status", "in_progress"),
                },
            }
            result = await self.interviews_collection.update_one(
                {"session_id": session_id},
                update_doc,
                upsert=True,
            )
            logger.info(
                f"Interview upsert result - matched: {result.matched_count}, modified: {result.modified_count}, upserted_id: {getattr(result, 'upserted_id', None)}"
            )
            return (result.modified_count > 0) or (getattr(result, "upserted_id", None) is not None)
        except Exception as e:
            logger.error(f"Failed to update interview: {e}")
            return False
    
    async def get_interview(self, session_id: str) -> Optional[InterviewModel]:
        """Get interview by session ID"""
        try:
            interview_data = await self.interviews_collection.find_one({"session_id": session_id})
            if interview_data:
                return InterviewModel(**interview_data)
            return None
        except Exception as e:
            logger.error(f"Failed to get interview: {e}")
            return None
    
    async def save_rules_collection(self, rules_data: RulesCollectionModel) -> str:
        """Save extracted rules collection to database (upsert if exists)"""
        try:
            rules_dict = rules_data.model_dump(by_alias=True, exclude={"id"})
            
            # Try to upsert (update if exists, insert if not)
            result = await self.rules_collections_collection.replace_one(
                {"interview_session_id": rules_data.interview_session_id},
                rules_dict,
                upsert=True
            )
            
            if result.upserted_id:
                logger.info(f"Rules collection inserted with ID: {result.upserted_id}")
                return str(result.upserted_id)
            else:
                logger.info(f"Rules collection updated for session: {rules_data.interview_session_id}")
                # Get the document ID for the updated record
                doc = await self.rules_collections_collection.find_one(
                    {"interview_session_id": rules_data.interview_session_id}
                )
                return str(doc["_id"]) if doc else "updated"
                
        except Exception as e:
            logger.error(f"Failed to save rules collection: {e}")
            raise
    
    async def save_individual_rules(self, session_id: str, rules: List[Dict[str, Any]]) -> List[str]:
        """Save individual rules to database (clear existing ones first)"""
        try:
            # First, delete any existing rules for this session
            delete_result = await self.rules_collection.delete_many({"interview_session_id": session_id})
            if delete_result.deleted_count > 0:
                logger.info(f"Deleted {delete_result.deleted_count} existing rules for session {session_id}")
            
            rule_ids = []
            for i, rule in enumerate(rules):
                # Generate a unique rule ID if not provided
                rule_id = rule.get("id", f"rule_{session_id}_{i}")
                
                rule_model = ExtractedRuleModel(
                    interview_session_id=session_id,
                    rule_id=rule_id,
                    trigger=rule.get("if", {}),  # Use "if" from the new format
                    action=rule.get("then", {}),  # Use "then" from the new format
                    priority=rule.get("priority", "medium"),
                    category=rule.get("category", "general"),
                    extracted_at=datetime.now(timezone.utc)
                )
                
                rule_dict = rule_model.model_dump(by_alias=True, exclude={"id"})
                result = await self.rules_collection.insert_one(rule_dict)
                rule_ids.append(str(result.inserted_id))
            
            logger.info(f"Saved {len(rule_ids)} individual rules for session {session_id}")
            return rule_ids
            
        except Exception as e:
            logger.error(f"Failed to save individual rules: {e}")
            raise
    
    async def get_rules_by_session(self, session_id: str) -> List[ExtractedRuleModel]:
        """Get all rules for a specific interview session"""
        try:
            cursor = self.rules_collection.find({"interview_session_id": session_id})
            rules = []
            async for rule_data in cursor:
                rules.append(ExtractedRuleModel(**rule_data))
            return rules
        except Exception as e:
            logger.error(f"Failed to get rules by session: {e}")
            return []
    
    async def get_all_interviews(self, limit: int = 50) -> List[InterviewModel]:
        """Get all interviews with pagination"""
        try:
            cursor = self.interviews_collection.find().sort("started_at", -1).limit(limit)
            interviews = []
            async for interview_data in cursor:
                interviews.append(InterviewModel(**interview_data))
            return interviews
        except Exception as e:
            logger.error(f"Failed to get interviews: {e}")
            return []
    
    async def get_all_rules(self, limit: int = 100) -> List[ExtractedRuleModel]:
        """Get all rules with pagination and retry logic"""
        for attempt in range(3):  # Try up to 3 times
            try:
                cursor = self.rules_collection.find().sort("extracted_at", -1).limit(limit)
                rules = []
                async for rule_data in cursor:
                    rules.append(ExtractedRuleModel(**rule_data))
                return rules
            except Exception as e:
                logger.error(f"Failed to get rules (attempt {attempt + 1}/3): {e}")
                if attempt < 2:  # Not the last attempt
                    await asyncio.sleep(1)  # Wait 1 second before retry
                    continue
                return []
    
    async def get_rules_collection(self, session_id: str) -> Optional[RulesCollectionModel]:
        """Get the rules collection for a session"""
        try:
            collection_data = await self.rules_collections_collection.find_one({
                "interview_session_id": session_id
            })
            if collection_data:
                return RulesCollectionModel(**collection_data)
            return None
        except Exception as e:
            logger.error(f"Failed to get rules collection: {e}")
            return None

# Global database manager instance
db_manager = DatabaseManager()
