import pandas as pd
from typing import List, Dict, Any
import os

class MetricsCollector:
    """
    Collects simulation metrics at each timestep and exports to CSV.
    """
    
    def __init__(self):
        self.records: List[Dict[str, Any]] = []
        self.prev_rate_limit: float | None = None

    def record(self, 
               timestamp: float, 
               throughput_mbps: float, 
               delay_ms: float, 
               drop_count: int, 
               queue_occupancy_bytes: int, 
               reward: float, 
               rate_limit_mbps: float):
        """
        Record a single timestep of simulation metrics.
        """
        if self.prev_rate_limit is None:
            rate_delta_mbps = 0.0
        else:
            rate_delta_mbps = abs(rate_limit_mbps - self.prev_rate_limit)
        
        self.prev_rate_limit = rate_limit_mbps

        self.records.append({
            "timestamp": float(timestamp),
            "throughput_mbps": float(throughput_mbps),
            "delay_ms": float(delay_ms),
            "drop_count": int(drop_count),
            "queue_occupancy_bytes": int(queue_occupancy_bytes),
            "reward": float(reward),
            "rate_limit_mbps": float(rate_limit_mbps),
            "rate_delta_mbps": float(rate_delta_mbps),
        })

    def to_dataframe(self) -> pd.DataFrame:
        """
        Convert collected records to a pandas DataFrame.
        """
        return pd.DataFrame(self.records)

    def export_csv(self, path: str):
        """
        Export collected records to a CSV file.
        """
        if not self.records:
            return

        df = self.to_dataframe()
        
        # Ensure the directory exists
        os.makedirs(os.path.dirname(os.path.abspath(path)), exist_ok=True)
        
        df.to_csv(path, index=False)

if __name__ == "__main__":
    # Simple standalone unit test
    print("Testing MetricsCollector...")
    collector = MetricsCollector()
    collector.record(1.0, 50.5, 12.0, 0, 1024, 0.95, 60.0)
    collector.record(2.0, 55.0, 15.0, 5, 2048, 0.80, 65.0)
    
    df = collector.to_dataframe()
    assert len(df) == 2, "DataFrame should have 2 rows"
    assert df.iloc[0]["rate_delta_mbps"] == 0.0, "First delta should be 0.0"
    assert df.iloc[1]["rate_delta_mbps"] == 5.0, "Second delta should be 5.0"
    
    test_path = "test_metrics.csv"
    collector.export_csv(test_path)
    assert os.path.exists(test_path), "CSV file was not created"
    os.remove(test_path)
    print("MetricsCollector tests passed!")
