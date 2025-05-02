# hmafqa/utils/token_counter.py
import tiktoken
from typing import Dict, List, Any, Optional

class TokenCounter:
    """
    Track token usage across the system.
    """
    
    def __init__(self):
        """Initialize the token counter."""
        self.total_prompt_tokens = 0
        self.total_completion_tokens = 0
        self.total_cost = 0
        self.calls_by_agent = {}
        
        # Approximate costs per 1K tokens (can be updated)
        self.costs = {
            "gpt-4": {"prompt": 0.03, "completion": 0.06},
            "gpt-4-turbo": {"prompt": 0.01, "completion": 0.03},
            "gpt-3.5-turbo": {"prompt": 0.0015, "completion": 0.002}
        }
    
    def count_tokens(self, text: str, model: str = "gpt-4") -> int:
        """Count the number of tokens in a text string."""
        try:
            # Get the encoder for the specified model
            if "gpt-4" in model:
                enc = tiktoken.encoding_for_model("gpt-4")
            elif "gpt-3.5" in model:
                enc = tiktoken.encoding_for_model("gpt-3.5-turbo")
            else:
                enc = tiktoken.get_encoding("cl100k_base")  # Default to cl100k
            
            # Count tokens
            token_count = len(enc.encode(text))
            return token_count
        except Exception as e:
            print(f"Error counting tokens: {e}")
            # Estimate 4 chars per token as fallback
            return len(text) // 4
    
    def track_usage(
        self, 
        prompt: str, 
        completion: str, 
        model: str, 
        agent_name: str
    ):
        """
        Track token usage for a single API call.
        
        Args:
            prompt: The prompt text
            completion: The completion text
            model: The model used
            agent_name: The name of the agent making the call
        """
        prompt_tokens = self.count_tokens(prompt, model)
        completion_tokens = self.count_tokens(completion, model)
        
        # Update totals
        self.total_prompt_tokens += prompt_tokens
        self.total_completion_tokens += completion_tokens
        
        # Calculate cost
        model_costs = self.costs.get(model, {"prompt": 0.01, "completion": 0.02})
        prompt_cost = (prompt_tokens / 1000) * model_costs["prompt"]
        completion_cost = (completion_tokens / 1000) * model_costs["completion"]
        
        # Update total cost
        self.total_cost += prompt_cost + completion_cost
        
        # Update agent stats
        if agent_name not in self.calls_by_agent:
            self.calls_by_agent[agent_name] = {
                "calls": 0,
                "prompt_tokens": 0,
                "completion_tokens": 0,
                "cost": 0
            }
        
        self.calls_by_agent[agent_name]["calls"] += 1
        self.calls_by_agent[agent_name]["prompt_tokens"] += prompt_tokens
        self.calls_by_agent[agent_name]["completion_tokens"] += completion_tokens
        self.calls_by_agent[agent_name]["cost"] += prompt_cost + completion_cost
    
    def get_summary(self) -> Dict[str, Any]:
        """Get a summary of token usage and costs."""
        return {
            "total_prompt_tokens": self.total_prompt_tokens,
            "total_completion_tokens": self.total_completion_tokens,
            "total_tokens": self.total_prompt_tokens + self.total_completion_tokens,
            "total_cost": round(self.total_cost, 4),
            "calls_by_agent": self.calls_by_agent
        }