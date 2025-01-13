from dataclasses import dataclass
from datetime import datetime
import json
from typing import Optional

@dataclass
class Trade:
    """
    Example class demonstrating JSON serialization/deserialization
    """
    symbol: str
    entry_price: float
    exit_price: Optional[float]
    entry_time: datetime
    exit_time: Optional[datetime]
    position_size: float
    side: str  # 'long' or 'short'
    
    def to_json(self) -> str:
        """
        Serialize the Trade object to JSON string
        """
        # Create a dictionary representation with datetime objects converted to ISO format
        trade_dict = {
            'symbol': self.symbol,
            'entry_price': self.entry_price,
            'exit_price': self.exit_price,
            'entry_time': self.entry_time.isoformat() if self.entry_time else None,
            'exit_time': self.exit_time.isoformat() if self.exit_time else None,
            'position_size': self.position_size,
            'side': self.side
        }
        return json.dumps(trade_dict)
    
    @classmethod
    def from_json(cls, json_str: str) -> 'Trade':
        """
        Create a Trade object from a JSON string
        """
        # Parse JSON string to dictionary
        trade_dict = json.loads(json_str)
        
        # Convert ISO format strings back to datetime objects
        entry_time = datetime.fromisoformat(trade_dict['entry_time']) if trade_dict['entry_time'] else None
        exit_time = datetime.fromisoformat(trade_dict['exit_time']) if trade_dict['exit_time'] else None
        
        # Create and return new Trade object
        return cls(
            symbol=trade_dict['symbol'],
            entry_price=trade_dict['entry_price'],
            exit_price=trade_dict['exit_price'],
            entry_time=entry_time,
            exit_time=exit_time,
            position_size=trade_dict['position_size'],
            side=trade_dict['side']
        )

# Example usage
if __name__ == "__main__":
    # Create a trade object
    trade = Trade(
        symbol="BTCUSDT",
        entry_price=50000.0,
        exit_price=None,  # Open trade
        entry_time=datetime.now(),
        exit_time=None,
        position_size=0.1,
        side="long"
    )
    
    # Serialize to JSON
    json_str = trade.to_json()
    print("Serialized trade:")
    print(json_str)
    print()
    
    # Deserialize back to Trade object
    new_trade = Trade.from_json(json_str)
    print("Deserialized trade:")
    print(f"Symbol: {new_trade.symbol}")
    print(f"Entry Price: {new_trade.entry_price}")
    print(f"Position Size: {new_trade.position_size}")
    print(f"Side: {new_trade.side}")
