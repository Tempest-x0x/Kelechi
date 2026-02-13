"""SIGHT Signal Generator - Complete signal generation with ICT + Confluence."""
from typing import Optional, List, Dict
import pandas as pd
import numpy as np
from datetime import datetime

from ..core.base import SignalGenerator
from ..core.types import (
    Timeframe, SignalType, TradeSetup, PairConfig, MarketBias
)
from .ict_engine import ICTEngine


class ICTSignalGenerator(SignalGenerator):
    """
    Complete signal generation combining ICT Engine (80%) and Confluence Filter (20%).
    
    Signal Generation Flow:
    1. ICT Engine generates base setup (80% weight)
    2. Confluence Filter validates exhaustion (20% weight)
    3. Combined validation determines final signal
    
    A signal is only generated when:
    - ICT criteria are fully met
    - At least one confluence filter triggers
    """
    
    def __init__(
        self,
        ict_engine: ICTEngine = None,
        confluence_filter = None  # Will be injected
    ):
        super().__init__("ICTSignalGenerator")
        self.ict_engine = ict_engine or ICTEngine()
        self.confluence_filter = confluence_filter
    
    def generate_signal(
        self,
        pair: str,
        htf_data: pd.DataFrame,
        ltf_data: pd.DataFrame,
        config: PairConfig
    ) -> Optional[TradeSetup]:
        """
        Generate trade signal from multi-timeframe data.
        
        Args:
            pair: Currency pair
            htf_data: Higher timeframe data (1H)
            ltf_data: Lower timeframe data (15m, 1m combined)
            config: Pair-specific configuration
            
        Returns:
            Validated TradeSetup or None
        """
        # Update ICT engine config
        self.ict_engine.update_config(config)
        
        # For this implementation, we expect:
        # - htf_data contains 1H data
        # - ltf_data is a dict with 15m and 1m data
        # Alternatively, we can split the data here
        
        # This is a simplified interface - in production, 
        # data would be properly structured
        self.log_warning("generate_signal requires structured multi-TF data")
        return None
    
    def generate_signal_mtf(
        self,
        pair: str,
        h1_data: pd.DataFrame,
        m15_data: pd.DataFrame,
        m1_data: pd.DataFrame,
        config: PairConfig
    ) -> Optional[TradeSetup]:
        """
        Generate trade signal from explicit multi-timeframe data.
        
        Signal Generation Process:
        1. ICT Engine Analysis (80% decision weight)
           - HTF bias alignment
           - Liquidity sweep detection
           - MSS confirmation
           - FVG entry identification
        
        2. Confluence Validation (20% decision weight)
           - Bollinger Band boundary touch
           - Keltner Channel boundary touch
        
        3. Final Validation
           - RR ratio check
           - Spread check
           - Daily limit check
        
        Args:
            pair: Currency pair
            h1_data: 1-Hour OHLCV data
            m15_data: 15-Minute OHLCV data
            m1_data: 1-Minute OHLCV data
            config: Pair-specific configuration
            
        Returns:
            Validated TradeSetup or None
        """
        # Update config
        self.ict_engine.update_config(config)
        
        # Step 1: Generate ICT setup
        setup = self.ict_engine.generate_setup(pair, h1_data, m15_data, m1_data)
        
        if setup is None:
            return None
        
        # Step 2: Apply confluence filter if available
        if self.confluence_filter is not None:
            bb_touch = self.confluence_filter.check_bollinger_touch(m15_data, setup.signal)
            kc_touch = self.confluence_filter.check_keltner_touch(m15_data, setup.signal)
            
            setup.bollinger_touch = bb_touch
            setup.keltner_touch = kc_touch
            
            # Require at least one confluence trigger
            if not bb_touch and not kc_touch:
                self.log_debug(f"{pair}: No confluence confirmation")
                setup.is_valid = False
                setup.invalidation_reasons.append("No confluence filter triggered")
                return setup
            
            # Calculate confluence score
            setup.confluence_score = self.confluence_filter.calculate_confluence_score(setup)
        
        # Step 3: Final validation
        is_valid, reasons = self.validate_setup(setup)
        setup.is_valid = is_valid
        setup.invalidation_reasons = reasons
        
        if is_valid:
            self.log_info(f"{pair}: Valid signal generated - {setup.signal.name}")
        
        return setup
    
    def validate_setup(self, setup: TradeSetup) -> tuple[bool, List[str]]:
        """
        Perform final validation on trade setup.
        
        Validation Criteria:
        1. ICT criteria met (via ICT engine)
        2. Confluence criteria met (BB or KC touch)
        3. RR ratio >= 2.2
        4. Signal direction matches HTF bias
        
        Args:
            setup: TradeSetup to validate
            
        Returns:
            Tuple of (is_valid, reasons)
        """
        reasons = []
        
        # Validate via ICT engine
        ict_valid, ict_reasons = self.ict_engine.validate_setup(setup)
        reasons.extend(ict_reasons)
        
        # Check confluence
        if not setup.bollinger_touch and not setup.keltner_touch:
            reasons.append("No confluence filter triggered")
        
        # Check RR ratio
        if setup.risk_reward_ratio < 2.2:
            reasons.append(f"RR ratio {setup.risk_reward_ratio:.2f} < 2.2")
        
        # Check direction alignment
        if setup.signal == SignalType.LONG and setup.htf_bias != MarketBias.BULLISH:
            reasons.append("LONG signal but HTF bias not bullish")
        if setup.signal == SignalType.SHORT and setup.htf_bias != MarketBias.BEARISH:
            reasons.append("SHORT signal but HTF bias not bearish")
        
        is_valid = len(reasons) == 0
        
        return (is_valid, reasons)
    
    def set_confluence_filter(self, confluence_filter) -> None:
        """Set the confluence filter instance."""
        self.confluence_filter = confluence_filter
        self.log_info("Confluence filter configured")


class SignalAggregator(SignalGenerator):
    """
    Aggregates signals across multiple pairs with prioritization.
    
    Prioritization Rules:
    1. Highest confluence score
    2. Highest RR ratio
    3. Lowest spread impact
    4. Most recent liquidity sweep
    """
    
    def __init__(self, signal_generators: Dict[str, ICTSignalGenerator] = None):
        super().__init__("SignalAggregator")
        self.generators = signal_generators or {}
        self.daily_signals: Dict[str, List[TradeSetup]] = {}
    
    def add_generator(self, pair: str, generator: ICTSignalGenerator) -> None:
        """Add a signal generator for a pair."""
        self.generators[pair] = generator
    
    def generate_signal(
        self,
        pair: str,
        htf_data: pd.DataFrame,
        ltf_data: pd.DataFrame,
        config: PairConfig
    ) -> Optional[TradeSetup]:
        """Generate signal for a specific pair."""
        if pair not in self.generators:
            self.log_warning(f"No generator configured for {pair}")
            return None
        
        return self.generators[pair].generate_signal(pair, htf_data, ltf_data, config)
    
    def generate_all_signals(
        self,
        data_provider,
        configs: Dict[str, PairConfig],
        start: datetime,
        end: datetime
    ) -> List[TradeSetup]:
        """
        Generate signals for all configured pairs.
        
        Args:
            data_provider: Data provider instance
            configs: Dict of pair configs
            start: Start datetime
            end: End datetime
            
        Returns:
            List of all valid setups, sorted by priority
        """
        all_setups = []
        
        for pair, config in configs.items():
            if pair not in self.generators:
                continue
            
            try:
                # Get multi-timeframe data
                h1_data = data_provider.get_ohlcv(pair, Timeframe.H1, start, end)
                m15_data = data_provider.get_ohlcv(pair, Timeframe.M15, start, end)
                m1_data = data_provider.get_ohlcv(pair, Timeframe.M1, start, end)
                
                # Generate signal
                setup = self.generators[pair].generate_signal_mtf(
                    pair, h1_data, m15_data, m1_data, config
                )
                
                if setup and setup.is_valid:
                    all_setups.append(setup)
                    
            except Exception as e:
                self.log_error(f"Error generating signal for {pair}: {e}")
        
        # Sort by priority
        all_setups.sort(key=lambda s: (
            -s.confluence_score,  # Higher confluence first
            -s.risk_reward_ratio, # Higher RR first
        ))
        
        return all_setups
    
    def validate_setup(self, setup: TradeSetup) -> bool:
        """Validate a setup from any generator."""
        return setup.is_valid
    
    def reset_daily_signals(self) -> None:
        """Reset daily signal tracking."""
        self.daily_signals = {pair: [] for pair in self.generators}
