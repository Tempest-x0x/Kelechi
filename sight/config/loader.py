"""SIGHT Config Loader - Load and manage configurations."""
import json
from pathlib import Path
from typing import Dict, Optional
from datetime import datetime

from ..core.types import PairConfig
from ..core.constants import SUPPORTED_PAIRS, CONFIG_DIR
from ..core.base import BaseLogger


class ConfigLoader(BaseLogger):
    """
    Configuration loader and manager.
    
    Handles:
    - Loading pair_config.json
    - Validating configurations
    - Providing defaults for missing pairs
    """
    
    def __init__(self, config_dir: str = CONFIG_DIR):
        super().__init__("ConfigLoader")
        self.config_dir = Path(config_dir)
        self.config_dir.mkdir(parents=True, exist_ok=True)
        self.pair_configs: Dict[str, PairConfig] = {}
    
    def load_pair_configs(self, filename: str = "pair_config.json") -> Dict[str, PairConfig]:
        """
        Load pair configurations from JSON file.
        
        Args:
            filename: Config filename
            
        Returns:
            Dict mapping pair to PairConfig
        """
        filepath = self.config_dir / filename
        
        if not filepath.exists():
            self.log_warning(f"Config file not found: {filepath}")
            return self._create_default_configs()
        
        try:
            with open(filepath, 'r') as f:
                data = json.load(f)
            
            for pair, config_data in data.get('pairs', {}).items():
                # Remove non-config fields
                config_data.pop('optimization_summary', None)
                
                try:
                    self.pair_configs[pair] = PairConfig.from_dict(config_data)
                except Exception as e:
                    self.log_warning(f"Invalid config for {pair}: {e}")
                    self.pair_configs[pair] = PairConfig(pair=pair)
            
            # Add defaults for missing pairs
            for pair in SUPPORTED_PAIRS:
                if pair not in self.pair_configs:
                    self.pair_configs[pair] = PairConfig(pair=pair)
            
            self.log_info(f"Loaded {len(self.pair_configs)} pair configs")
            
        except Exception as e:
            self.log_error(f"Error loading config: {e}")
            return self._create_default_configs()
        
        return self.pair_configs
    
    def _create_default_configs(self) -> Dict[str, PairConfig]:
        """Create default configurations for all pairs."""
        for pair in SUPPORTED_PAIRS:
            self.pair_configs[pair] = PairConfig(pair=pair)
        return self.pair_configs
    
    def save_configs(
        self,
        configs: Dict[str, PairConfig] = None,
        filename: str = "pair_config.json"
    ) -> Path:
        """
        Save configurations to JSON file.
        
        Args:
            configs: Configurations to save (default: current)
            filename: Output filename
            
        Returns:
            Path to saved file
        """
        configs = configs or self.pair_configs
        
        output = {
            "generated": datetime.now().isoformat(),
            "version": "1.0",
            "pairs": {
                pair: config.to_dict() 
                for pair, config in configs.items()
            }
        }
        
        filepath = self.config_dir / filename
        with open(filepath, 'w') as f:
            json.dump(output, f, indent=2)
        
        self.log_info(f"Saved configs to {filepath}")
        return filepath
    
    def get_config(self, pair: str) -> PairConfig:
        """Get config for a pair, with default fallback."""
        if pair in self.pair_configs:
            return self.pair_configs[pair]
        return PairConfig(pair=pair)
    
    def update_config(self, pair: str, config: PairConfig) -> None:
        """Update configuration for a pair."""
        self.pair_configs[pair] = config
    
    def validate_configs(self) -> Dict[str, bool]:
        """
        Validate all configurations.
        
        Returns:
            Dict mapping pair to validation status
        """
        results = {}
        
        for pair, config in self.pair_configs.items():
            is_valid = (
                config.sweep_depth_atr_multiple > 0 and
                config.displacement_threshold_atr > 0 and
                0 <= config.fvg_entry_offset <= 1 and
                config.htf_ema_period > 0 and
                config.default_risk_reward >= 1.0
            )
            results[pair] = is_valid
            
            if not is_valid:
                self.log_warning(f"Invalid config for {pair}")
        
        return results


def generate_example_config() -> Dict:
    """Generate example pair_config.json structure."""
    return {
        "generated": datetime.now().isoformat(),
        "version": "1.0",
        "pairs": {
            "EURUSD": {
                "pair": "EURUSD",
                "sweep_depth_atr_multiple": 0.5,
                "sweep_lookback_candles": 48,
                "displacement_threshold_atr": 1.5,
                "displacement_body_percent_min": 0.7,
                "fvg_min_size_atr": 0.3,
                "fvg_entry_offset": 0.5,
                "htf_ema_period": 200,
                "swing_strength": 3,
                "default_risk_reward": 2.2,
                "max_spread_percent_of_target": 0.15,
                "max_daily_trades": 2,
                "bb_period": 20,
                "bb_std": 2.0,
                "kc_period": 20,
                "kc_atr_multiple": 1.5,
                "backtest_win_rate": 0.72,
                "backtest_profit_factor": 2.4,
                "validation_passed": True,
                "optimization_iterations": 45
            }
        }
    }
