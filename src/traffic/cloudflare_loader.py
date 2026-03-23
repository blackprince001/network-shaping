import json
import os
from typing import List


def load_cloudflare_traffic(
    json_path: str,
    scale_mbps: float = 80.0,
) -> List[float]:
    """
    Parse a Cloudflare Radar JSON file and return a list of demand values
    in Mbps, one per hourly timestamp.

    Args:
        json_path: Path to the cloudflare JSON file.
        scale_mbps: Multiply normalized values by this to get Mbps demand.

    Returns:
        List of demand values in Mbps, indexed by simulation step.
    """
    if not os.path.exists(json_path):
        raise FileNotFoundError(f"Cloudflare traffic file not found: {json_path}")

    with open(json_path, "r") as f:
        data = json.load(f)

    result = data.get("result", {})
    main = result.get("main", {})
    values_raw = main.get("values", [])

    if not values_raw:
        raise ValueError(f"No 'values' found in {json_path}")

    demands = [float(v) * scale_mbps for v in values_raw]
    return demands


def get_traffic_at_step(
    demands: List[float],
    step: int,
) -> float:
    """
    Get demand for a specific simulation step.
    Cycles through the demand profile if steps exceed available data.
    """
    if not demands:
        return 0.0
    return demands[step % len(demands)]


if __name__ == "__main__":
    import sys
    sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
    
    json_path = os.path.join(
        os.path.dirname(os.path.dirname(os.path.abspath(__file__))),
        "cloudflare_hourly.json"
    )
    
    if os.path.exists(json_path):
        demands = load_cloudflare_traffic(json_path, scale_mbps=80.0)
        print(f"Loaded {len(demands)} hourly demand values")
        print(f"Min: {min(demands):.2f} Mbps, Max: {max(demands):.2f} Mbps")
        print(f"First 24 values: {[f'{d:.1f}' for d in demands[:24]]}")
        
        # Test cycling
        step_demand = get_traffic_at_step(demands, 170)
        print(f"Step 170 (cycles to step 2): {step_demand:.2f} Gbps")
    else:
        print(f"Cloudflare file not found at {json_path}, skipping test")
