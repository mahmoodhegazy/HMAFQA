# hmafqa/conversation.py
from typing import Dict, List, Any, Optional
import time
import uuid

class Conversation:
    """
    Manage conversation history and context for multi-turn interactions.
    """
    
    def __init__(self, conversation_id: Optional[str] = None):
        """
        Initialize a conversation.
        
        Args:
            conversation_id: Optional ID for the conversation
        """
        self.conversation_id = conversation_id or str(uuid.uuid4())
        self.messages = []
        self.created_at = time.time()
        self.last_updated = self.created_at
        self.metadata = {}
    
    def add_message(self, role: str, content: str, metadata: Optional[Dict[str, Any]] = None):
        """
        Add a message to the conversation.
        
        Args:
            role: The role of the message sender ("user" or "assistant")
            content: The message content
            metadata: Optional metadata for the message
        """
        message = {
            "id": str(uuid.uuid4()),
            "role": role,
            "content": content,
            "timestamp": time.time(),
            "metadata": metadata or {}
        }
        
        self.messages.append(message)
        self.last_updated = message["timestamp"]
    
    def get_history(self, max_messages: Optional[int] = None) -> List[Dict[str, Any]]:
        """
        Get the conversation history.
        
        Args:
            max_messages: Maximum number of messages to retrieve
            
        Returns:
            List of messages
        """
        if max_messages is not None:
            return self.messages[-max_messages:]
        
        return self.messages.copy()
    
    def get_context(self) -> Dict[str, Any]:
        """
        Get the conversation context.
        
        Returns:
            Dictionary with conversation context
        """
        return {
            "conversation_id": self.conversation_id,
            "message_count": len(self.messages),
            "duration": time.time() - self.created_at,
            "metadata": self.metadata,
            "last_messages": self.get_history(3)  # Last 3 messages for context
        }
    
    def to_dict(self) -> Dict[str, Any]:
        """
        Convert the conversation to a dictionary.
        
        Returns:
            Dictionary representation of the conversation
        """
        return {
            "conversation_id": self.conversation_id,
            "messages": self.messages,
            "created_at": self.created_at,
            "last_updated": self.last_updated,
            "metadata": self.metadata
        }
    
    @classmethod
    def from_dict(cls, data: Dict[str, Any]) -> 'Conversation':
        """
        Create a conversation from a dictionary.
        
        Args:
            data: Dictionary representation of a conversation
            
        Returns:
            Conversation instance
        """
        conversation = cls(conversation_id=data.get("conversation_id"))
        conversation.messages = data.get("messages", [])
        conversation.created_at = data.get("created_at", time.time())
        conversation.last_updated = data.get("last_updated", time.time())
        conversation.metadata = data.get("metadata", {})
        
        return conversation