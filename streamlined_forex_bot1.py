#!/usr/bin/env python3
"""
STREAMLINED 50/50 FOREX SIGNAL BOT - MAIN BOT FILE
This file contains all the bot logic. Call main() to start the bot.
"""

import os
import json
import requests
import talib
import numpy as np
from datetime import datetime, timedelta
from telegram import Update
from telegram.ext import Application, CommandHandler, ContextTypes
import asyncio
import pytz
import logging

# ================================================
# CONFIGURATION
# ================================================
class Config:
    # API Keys
    TELEGRAM_BOT_TOKEN = os.getenv('TELEGRAM_BOT_TOKEN', 'your_telegram_token')
    TWELVE_DATA_API_KEY = os.getenv('TWELVE_DATA_KEY', 'your_twelve_data_key')
    DEEPSEEK_API_KEY = os.getenv('DEEPSEEK_API_KEY', 'your_deepseek_key')
    
    # Currency Pairs
    CURRENCY_PAIRS = {
        'EURUSD': '🇪🇺 EUR/USD 🇺🇸',
        'USDJPY': '🇺🇸 USD/JPY 🇯🇵',
        'GBPUSD': '🇬🇧 GBP/USD 🇺🇸',
        'AUDUSD': '🇦🇺 AUD/USD 🇺🇸',
        'USDCAD': '🇺🇸 USD/CAD 🇨🇦',
        'EURGBP': '🇪🇺 EUR/GBP 🇬🇧',
        'GBPJPY': '🇬🇧 GBP/JPY 🇯🇵'
    }
    
    # Trading Settings
    MINIMUM_CONFIDENCE = 65
    AUTO_SCAN_ENABLED = True
    AUTO_SCAN_INTERVAL = 30
    AUTO_SCAN_PAIRS = ['EURUSD', 'GBPUSD', 'USDJPY']
    MAX_SIGNALS_PER_HOUR = 8
    ENABLE_NOTIFICATIONS = True
    AUTO_ENABLE_FOR_NEW_USERS = True

# ================================================
# MARKET HOURS TRACKER
# ================================================
class MarketHoursTracker:
    SESSIONS = {
        'sydney': {'open': 22, 'close': 7, 'pairs': ['AUDUSD'], 'name': 'Sydney 🌏'},
        'tokyo': {'open': 0, 'close': 9, 'pairs': ['USDJPY', 'GBPJPY'], 'name': 'Tokyo 🇯🇵'},
        'london': {'open': 8, 'close': 17, 'pairs': ['EURUSD', 'GBPUSD', 'EURGBP'], 'name': 'London 🇬🇧'},
        'new_york': {'open': 13, 'close': 22, 'pairs': ['EURUSD', 'GBPUSD', 'USDCAD'], 'name': 'New York 🇺🇸'}
    }
    
    @staticmethod
    def is_forex_market_open():
        now_utc = datetime.now(pytz.UTC)
        day_of_week = now_utc.weekday()
        hour_utc = now_utc.hour
        
        if day_of_week == 6 and hour_utc < 22:
            return False
        if day_of_week == 5 and hour_utc >= 22:
            return False
        
        return True
    
    @staticmethod
    def get_active_sessions():
        now_utc = datetime.now(pytz.UTC)
        hour_utc = now_utc.hour
        active = []
        
        for session_name, times in MarketHoursTracker.SESSIONS.items():
            open_hour = times['open']
            close_hour = times['close']
            
            if open_hour > close_hour:
                if hour_utc >= open_hour or hour_utc < close_hour:
                    active.append(session_name)
            else:
                if open_hour <= hour_utc < close_hour:
                    active.append(session_name)
        
        return active
    
    @staticmethod
    def get_optimal_pairs():
        if not MarketHoursTracker.is_forex_market_open():
            return []
        
        active_sessions = MarketHoursTracker.get_active_sessions()
        optimal_pairs = set()
        
        for session_name in active_sessions:
            pairs = MarketHoursTracker.SESSIONS[session_name]['pairs']
            optimal_pairs.update(pairs)
        
        return list(optimal_pairs)

# ================================================
# TECHNICAL ANALYZER (TA-Lib)
# ================================================
class TechnicalAnalyzer:
    def calculate_all_indicators(self, ohlc_data):
        """Calculate comprehensive technical indicators with TA-Lib"""
        
        values = ohlc_data['values']
        closes = np.array([float(v['close']) for v in values])
        opens = np.array([float(v['open']) for v in values])
        highs = np.array([float(v['high']) for v in values])
        lows = np.array([float(v['low']) for v in values])
        
        if len(closes) < 50:
            return self._get_basic_indicators(closes, highs, lows)
        
        indicators = {
            'trend': self._calculate_trend(closes, highs, lows),
            'momentum': self._calculate_momentum(closes, highs, lows),
            'volatility': self._calculate_volatility(closes, highs, lows),
            'levels': self._calculate_levels(closes, highs, lows),
            'patterns': self._detect_patterns(opens, highs, lows, closes)
        }
        
        # Calculate overall score
        indicators['score'] = self._calculate_score(indicators, closes[-1])
        
        return indicators
    
    def _calculate_trend(self, closes, highs, lows):
        return {
            'ema_9': talib.EMA(closes, 9)[-1],
            'ema_21': talib.EMA(closes, 21)[-1],
            'ema_50': talib.EMA(closes, 50)[-1] if len(closes) >= 50 else None,
            'sma_20': talib.SMA(closes, 20)[-1],
            'adx': talib.ADX(highs, lows, closes, 14)[-1],
            'plus_di': talib.PLUS_DI(highs, lows, closes, 14)[-1],
            'minus_di': talib.MINUS_DI(highs, lows, closes, 14)[-1]
        }
    
    def _calculate_momentum(self, closes, highs, lows):
        macd, signal, hist = talib.MACD(closes, 12, 26, 9)
        stoch_k, stoch_d = talib.STOCH(highs, lows, closes, 14, 3, 3)
        
        return {
            'rsi': talib.RSI(closes, 14)[-1],
            'rsi_7': talib.RSI(closes, 7)[-1],
            'macd': macd[-1],
            'macd_signal': signal[-1],
            'macd_hist': hist[-1],
            'stoch_k': stoch_k[-1],
            'stoch_d': stoch_d[-1],
            'williams_r': talib.WILLR(highs, lows, closes, 14)[-1]
        }
    
    def _calculate_volatility(self, closes, highs, lows):
        bb_upper, bb_middle, bb_lower = talib.BBANDS(closes, 20, 2, 2)
        atr = talib.ATR(highs, lows, closes, 14)[-1]
        
        return {
            'bb_upper': bb_upper[-1],
            'bb_middle': bb_middle[-1],
            'bb_lower': bb_lower[-1],
            'atr': atr,
            'atr_percent': atr / closes[-1]
        }
    
    def _calculate_levels(self, closes, highs, lows):
        # Pivot points
        pivot = (highs[-2] + lows[-2] + closes[-2]) / 3 if len(closes) >= 2 else closes[-1]
        r1 = 2 * pivot - lows[-2] if len(lows) >= 2 else pivot
        s1 = 2 * pivot - highs[-2] if len(highs) >= 2 else pivot
        
        return {
            'support': s1,
            'resistance': r1,
            'pivot': pivot
        }
    
    def _detect_patterns(self, opens, highs, lows, closes):
        patterns = []
        
        # Bullish patterns
        if talib.CDLHAMMER(opens, highs, lows, closes)[-1] > 0:
            patterns.append('HAMMER_BULLISH')
        if talib.CDLENGULFING(opens, highs, lows, closes)[-1] > 0:
            patterns.append('ENGULFING_BULLISH')
        if talib.CDLMORNINGSTAR(opens, highs, lows, closes)[-1] > 0:
            patterns.append('MORNING_STAR_BULLISH')
        
        # Bearish patterns
        if talib.CDLSHOOTINGSTAR(opens, highs, lows, closes)[-1] < 0:
            patterns.append('SHOOTING_STAR_BEARISH')
        if talib.CDLENGULFING(opens, highs, lows, closes)[-1] < 0:
            patterns.append('ENGULFING_BEARISH')
        if talib.CDLEVENINGSTAR(opens, highs, lows, closes)[-1] < 0:
            patterns.append('EVENING_STAR_BEARISH')
        
        return patterns
    
    def _calculate_score(self, indicators, current_price):
        score = 0
        
        # Trend score (30 points)
        trend = indicators['trend']
        if trend['ema_9'] > trend['ema_21']:
            score += 15
        if trend['adx'] > 25:
            score += 10
        if trend['plus_di'] > trend['minus_di']:
            score += 5
        
        # Momentum score (30 points)
        momentum = indicators['momentum']
        if momentum['rsi'] < 40 or momentum['rsi'] > 60:
            score += 15
        if momentum['macd'] > momentum['macd_signal']:
            score += 10
        if momentum['stoch_k'] < 20 or momentum['stoch_k'] > 80:
            score += 5
        
        # Volatility score (20 points)
        volatility = indicators['volatility']
        bb_range = volatility['bb_upper'] - volatility['bb_lower']
        if bb_range > 0:
            bb_position = (current_price - volatility['bb_lower']) / bb_range
            if bb_position < 0.2 or bb_position > 0.8:
                score += 15
        if volatility['atr_percent'] < 0.002:
            score += 5
        
        # Pattern score (20 points)
        if indicators['patterns']:
            score += min(len(indicators['patterns']) * 5, 20)
        
        return score
    
    def _get_basic_indicators(self, closes, highs, lows):
        """Fallback for insufficient data"""
        return {
            'trend': {'ema_9': closes[-1], 'ema_21': closes[-1], 'adx': 20},
            'momentum': {'rsi': 50, 'macd': 0, 'macd_signal': 0},
            'volatility': {'atr': 0.001, 'atr_percent': 0.001},
            'levels': {'support': lows[-1], 'resistance': highs[-1]},
            'patterns': [],
            'score': 50
        }
    
    def determine_direction(self, indicators, current_price):
        """Determine trade direction"""
        buy_signals = 0
        sell_signals = 0
        
        # Trend
        trend = indicators['trend']
        if trend['ema_9'] > trend['ema_21']:
            buy_signals += 3
        else:
            sell_signals += 3
        
        # Momentum
        momentum = indicators['momentum']
        if momentum['rsi'] < 40:
            buy_signals += 3
        elif momentum['rsi'] > 60:
            sell_signals += 3
        
        if momentum['macd'] > momentum['macd_signal']:
            buy_signals += 2
        else:
            sell_signals += 2
        
        # Patterns
        for pattern in indicators['patterns']:
            if 'BULLISH' in pattern:
                buy_signals += 1
            elif 'BEARISH' in pattern:
                sell_signals += 1
        
        return 'CALL' if buy_signals > sell_signals else 'PUT'

# ================================================
# DEEPSEEK AI CLIENT
# ================================================
class DeepSeekClient:
    def __init__(self, api_key):
        self.api_key = api_key
        self.cache = {}
    
    async def analyze_with_context(self, symbol, bot_analysis):
        """Get AI insights to complement bot analysis"""
        
        prompt = f"""You are providing ADDITIONAL INSIGHTS to complement a trading bot's analysis.

SYMBOL: {symbol}
CURRENT PRICE: {bot_analysis['current_price']:.5f}

BOT'S ANALYSIS:
- Direction: {bot_analysis['direction']}
- Confidence: {bot_analysis['confidence']:.1f}%
- Technical Score: {bot_analysis['score']}/100
- Risk: {bot_analysis['risk_level']}

KEY INDICATORS:
- RSI: {bot_analysis['indicators']['momentum']['rsi']:.1f}
- MACD: {bot_analysis['indicators']['momentum']['macd']:.6f}
- ADX: {bot_analysis['indicators']['trend']['adx']:.1f}
- ATR%: {bot_analysis['indicators']['volatility']['atr_percent']*100:.2f}%

Provide brief insights in JSON:
{{
    "sentiment": "BULLISH/BEARISH/NEUTRAL",
    "risk_factors": ["factor1", "factor2"],
    "confidence_adjustment": 0.8-1.2,
    "key_insight": "one sentence insight"
}}"""
        
        try:
            response = requests.post(
                'https://api.deepseek.com/v1/chat/completions',
                headers={
                    'Authorization': f'Bearer {self.api_key}',
                    'Content-Type': 'application/json'
                },
                json={
                    'model': 'deepseek-chat',
                    'messages': [
                        {'role': 'system', 'content': 'You are a forex market analyst. Be concise.'},
                        {'role': 'user', 'content': prompt}
                    ],
                    'temperature': 0.3,
                    'max_tokens': 300
                },
                timeout=20
            )
            
            if response.status_code == 200:
                content = response.json()['choices'][0]['message']['content']
                return self._parse_ai_response(content)
        
        except Exception as e:
            print(f"AI error: {e}")
        
        return self._get_default_response()
    
    def _parse_ai_response(self, content):
        try:
            json_start = content.find('{')
            json_end = content.rfind('}') + 1
            if json_start != -1 and json_end > json_start:
                return json.loads(content[json_start:json_end])
        except:
            pass
        return self._get_default_response()
    
    def _get_default_response(self):
        return {
            'sentiment': 'NEUTRAL',
            'risk_factors': ['Market volatility'],
            'confidence_adjustment': 1.0,
            'key_insight': 'Standard market conditions'
        }

# ================================================
# DECISION ENGINE (50/50 Bot-AI)
# ================================================
class DecisionEngine:
    def __init__(self):
        self.technical_analyzer = TechnicalAnalyzer()
        self.deepseek = DeepSeekClient(Config.DEEPSEEK_API_KEY)
    
    async def analyze_50_50(self, symbol, ohlc_data):
        """50/50 Bot-AI analysis pipeline"""
        
        print(f"🔍 Starting 50/50 analysis for {symbol}")
        
        # PHASE 1: Bot Technical Analysis (50%)
        print("🤖 Phase 1: Bot technical analysis...")
        indicators = self.technical_analyzer.calculate_all_indicators(ohlc_data)
        
        # Get current price
        current_price = float(ohlc_data['values'][0]['close'])
        
        # Determine direction
        direction = self.technical_analyzer.determine_direction(indicators, current_price)
        
        # Calculate bot confidence (50% weight)
        bot_confidence = indicators['score']
        
        if bot_confidence < 40:
            print(f"❌ Bot rejection: Score {bot_confidence}")
            return None
        
        # Build bot analysis
        bot_analysis = {
            'symbol': symbol,
            'direction': direction,
            'confidence': bot_confidence,
            'current_price': current_price,
            'indicators': indicators,
            'score': indicators['score'],
            'risk_level': self._assess_risk(indicators)
        }
        
        # PHASE 2: AI Analysis (50%)
        print("🧠 Phase 2: AI analysis...")
        ai_insights = await self.deepseek.analyze_with_context(symbol, bot_analysis)
        
        # PHASE 3: Combine 50/50
        print("⚖️ Phase 3: Combining decisions...")
        final_signal = self._combine_50_50(bot_analysis, ai_insights)
        
        # PHASE 4: Validation
        if not self._validate_signal(final_signal):
            return None
        
        print(f"✅ Signal approved: {final_signal['confidence']:.1f}%")
        return final_signal
    
    def _assess_risk(self, indicators):
        volatility = indicators['volatility']['atr_percent']
        
        if volatility > 0.002:
            return 'HIGH'
        elif volatility > 0.001:
            return 'MEDIUM'
        else:
            return 'LOW'
    
    def _combine_50_50(self, bot_analysis, ai_insights):
        """Combine bot and AI with 50/50 weight"""
        
        # Bot contributes 50%
        bot_weighted = bot_analysis['confidence'] * 0.5
        
        # AI adjustment contributes up to 50%
        ai_adjustment = ai_insights.get('confidence_adjustment', 1.0)
        ai_weighted = (ai_adjustment - 0.8) * 125  # Convert 0.8-1.2 to 0-50
        
        # Final confidence
        final_confidence = bot_weighted + ai_weighted
        final_confidence = max(0, min(100, final_confidence))
        
        return {
            **bot_analysis,
            'confidence': final_confidence,
            'ai_insights': ai_insights,
            'decision_source': '50_50_BOT_AI'
        }
    
    def _validate_signal(self, signal):
        """Final validation checks"""
        
        # Minimum confidence
        if signal['confidence'] < Config.MINIMUM_CONFIDENCE:
            print(f"❌ Confidence too low: {signal['confidence']:.1f}%")
            return False
        
        # Extreme volatility check
        if signal['indicators']['volatility']['atr_percent'] > 0.003:
            print("❌ Volatility too high")
            return False
        
        # RSI extreme check
        rsi = signal['indicators']['momentum']['rsi']
        if rsi < 10 or rsi > 90:
            print(f"❌ RSI extreme: {rsi:.1f}")
            return False
        
        return True

# ================================================
# TWELVE DATA CLIENT
# ================================================
class TwelveDataClient:
    def __init__(self, api_key):
        self.api_key = api_key
        self.base_url = 'https://api.twelvedata.com'
    
    def get_time_series(self, symbol, interval='5min', outputsize=100):
        url = f'{self.base_url}/time_series'
        params = {
            'symbol': symbol,
            'interval': interval,
            'outputsize': outputsize,
            'apikey': self.api_key
        }
        response = requests.get(url, params=params, timeout=10)
        return response.json()

# ================================================
# SIGNAL FORMATTER
# ================================================
class SignalFormatter:
    @staticmethod
    def format_signal(signal, display_name):
        """Format signal for Telegram"""
        
        now = datetime.now(pytz.UTC)
        entry_time = now + timedelta(minutes=2)
        expiry_time = entry_time + timedelta(minutes=5)
        
        direction_emoji = '🟢📈' if signal['direction'] == 'CALL' else '🔴📉'
        confidence = signal['confidence']
        
        if confidence >= 85:
            conf_emoji = '🔥🔥🔥'
        elif confidence >= 75:
            conf_emoji = '🔥🔥'
        else:
            conf_emoji = '🔥'
        
        risk_emoji = {'HIGH': '🔴', 'MEDIUM': '🟡', 'LOW': '🟢'}[signal['risk_level']]
        
        ai_insight = signal['ai_insights'].get('key_insight', 'Analysis complete')
        
        return f"""🤖 **50/50 AI-BOT SIGNAL**

🎯 **TRADE DETAILS**
• Pair: {display_name}
• Direction: {signal['direction']} {direction_emoji}
• Entry: {entry_time.strftime('%H:%M UTC')}
• Expiry: 5 minutes

📊 **ANALYSIS**
• Confidence: {confidence:.1f}% {conf_emoji}
• Risk: {signal['risk_level']} {risk_emoji}
• Bot Score: {signal['score']}/100
• AI Insight: {ai_insight}

🎯 **KEY LEVELS**
• Support: {signal['indicators']['levels']['support']:.5f}
• Resistance: {signal['indicators']['levels']['resistance']:.5f}
• Current: {signal['current_price']:.5f}

📊 **INDICATORS**
• RSI: {signal['indicators']['momentum']['rsi']:.1f}
• MACD: {'BULLISH' if signal['indicators']['momentum']['macd'] > signal['indicators']['momentum']['macd_signal'] else 'BEARISH'}
• ADX: {signal['indicators']['trend']['adx']:.1f}

⏰ Execute within 2 minutes on your platform!
⚠️ Max 2% of balance recommended"""

# ================================================
# USER PREFERENCES
# ================================================
class UserPreferences:
    def __init__(self):
        self.user_settings = {}
    
    def enable_auto_signals(self, chat_id, pairs, interval):
        self.user_settings[chat_id] = {
            'auto_enabled': True,
            'pairs': pairs,
            'interval': interval,
            'last_signal_time': {},
            'signals_this_hour': 0,
            'hour_reset_time': datetime.now(pytz.UTC)
        }
    
    def disable_auto_signals(self, chat_id):
        if chat_id in self.user_settings:
            self.user_settings[chat_id]['auto_enabled'] = False
    
    def is_auto_enabled(self, chat_id):
        return self.user_settings.get(chat_id, {}).get('auto_enabled', False)
    
    def get_user_pairs(self, chat_id):
        return self.user_settings.get(chat_id, {}).get('pairs', Config.AUTO_SCAN_PAIRS)
    
    def can_send_signal(self, chat_id, pair):
        if chat_id not in self.user_settings:
            return True
        
        settings = self.user_settings[chat_id]
        now = datetime.now(pytz.UTC)
        
        # Reset hourly counter
        if now - settings['hour_reset_time'] > timedelta(hours=1):
            settings['signals_this_hour'] = 0
            settings['hour_reset_time'] = now
        
        # Check hourly limit
        if settings['signals_this_hour'] >= Config.MAX_SIGNALS_PER_HOUR:
            return False
        
        # Check per-pair cooldown
        last_signal = settings['last_signal_time'].get(pair)
        if last_signal:
            minutes_since = (now - last_signal).total_seconds() / 60
            if minutes_since < settings['interval']:
                return False
        
        return True
    
    def record_signal_sent(self, chat_id, pair):
        if chat_id in self.user_settings:
            settings = self.user_settings[chat_id]
            settings['signals_this_hour'] += 1
            settings['last_signal_time'][pair] = datetime.now(pytz.UTC)

# ================================================
# MAIN BOT
# ================================================
class ForexSignalBot:
    def __init__(self):
        self.twelve_data = TwelveDataClient(Config.TWELVE_DATA_API_KEY)
        self.decision_engine = DecisionEngine()
        self.user_prefs = UserPreferences()
        self.notification_sent = {}
    
    async def analyze_pair(self, symbol):
        """Analyze pair with 50/50 method"""
        
        is_optimal, reason = MarketHoursTracker.is_forex_market_open(), "Market open"
        
        if not is_optimal:
            return {'error': 'Market closed'}, None
        
        try:
            ohlc_data = self.twelve_data.get_time_series(symbol, '5min', 100)
            
            if 'values' not in ohlc_data or len(ohlc_data['values']) < 50:
                return {'error': 'Insufficient data'}, None
            
            signal = await self.decision_engine.analyze_50_50(symbol, ohlc_data)
            
            return signal if signal else {'error': 'No valid signal'}, None
            
        except Exception as e:
            return {'error': str(e)}, None

# ================================================
# TELEGRAM HANDLERS
# ================================================
bot_instance = None
user_chat_ids = set()

async def start(update: Update, context: ContextTypes.DEFAULT_TYPE):
    chat_id = update.effective_chat.id
    user_chat_ids.add(chat_id)
    
    if Config.AUTO_ENABLE_FOR_NEW_USERS:
        bot_instance.user_prefs.enable_auto_signals(
            chat_id, Config.AUTO_SCAN_PAIRS, Config.AUTO_SCAN_INTERVAL
        )
    
    await update.message.reply_text(f"""🤖 **50/50 AI-BOT SIGNAL BOT**

✅ Auto-signals: ENABLED
📊 Pairs: {', '.join(Config.AUTO_SCAN_PAIRS)}
⏱️ Interval: Every {Config.AUTO_SCAN_INTERVAL} min

Commands:
/analyze [PAIR] - Get instant signal
/pairs - List pairs
/status - Bot status
/auto_off - Disable auto-signals

You'll receive signals automatically!""", parse_mode='Markdown')

async def analyze(update: Update, context: ContextTypes.DEFAULT_TYPE):
    if not context.args:
        await update.message.reply_text("Usage: /analyze EURUSD")
        return
    
    symbol = context.args[0].upper()
    
    if symbol not in Config.CURRENCY_PAIRS:
        await update.message.reply_text("Invalid pair. Use /pairs")
        return
    
    if not MarketHoursTracker.is_forex_market_open():
        await update.message.reply_text("❌ Market closed")
        return
    
    await update.message.reply_text(f"🔍 Analyzing {symbol}...")
    
    try:
        signal, _ = await bot_instance.analyze_pair(symbol)
        
        if not signal or 'error' in signal:
            await update.message.reply_text(f"❌ {signal.get('error', 'Analysis failed')}")
            return
        
        if signal['confidence'] < Config.MINIMUM_CONFIDENCE:
            await update.message.reply_text(
                f"⚠️ Low confidence: {signal['confidence']:.1f}%"
            )
            return
        
        display_name = Config.CURRENCY_PAIRS[symbol]
        message = SignalFormatter.format_signal(signal, display_name)
        
        await update.message.reply_text(message, parse_mode='Markdown')
        
    except Exception as e:
        await update.message.reply_text(f"❌ Error: {e}")

async def pairs(update: Update, context: ContextTypes.DEFAULT_TYPE):
    pairs_list = '\n'.join([f"• {code}: {name}" for code, name in Config.CURRENCY_PAIRS.items()])
    await update.message.reply_text(f"**Available Pairs:**\n\n{pairs_list}", parse_mode='Markdown')

async def status(update: Update, context: ContextTypes.DEFAULT_TYPE):
    chat_id = update.effective_chat.id
    auto_status = "✅ ENABLED" if bot_instance.user_prefs.is_auto_enabled(chat_id) else "⏸️ DISABLED"
    
    await update.message.reply_text(f"""**Bot Status:**

Auto-Signals: {auto_status}
Min Confidence: {Config.MINIMUM_CONFIDENCE}%
Scan Interval: {Config.AUTO_SCAN_INTERVAL} min
Market: {'🟢 OPEN' if MarketHoursTracker.is_forex_market_open() else '🔴 CLOSED'}""", parse_mode='Markdown')

async def auto_off(update: Update, context: ContextTypes.DEFAULT_TYPE):
    bot_instance.user_prefs.disable_auto_signals(update.effective_chat.id)
    await update.message.reply_text("⏸️ Auto-signals disabled")

async def auto_scan_markets(context: ContextTypes.DEFAULT_TYPE):
    """Automatically scan optimal pairs and send signals"""
    
    if not MarketHoursTracker.is_forex_market_open():
        return
    
    optimal_pairs = MarketHoursTracker.get_optimal_pairs()
    
    if not optimal_pairs:
        return
    
    print(f"🔍 Auto-scanning {len(optimal_pairs)} optimal pairs...")
    
    for chat_id in list(user_chat_ids):
        if not bot_instance.user_prefs.is_auto_enabled(chat_id):
            continue
        
        user_pairs = bot_instance.user_prefs.get_user_pairs(chat_id)
        pairs_to_scan = [p for p in optimal_pairs if p in user_pairs]
        
        for symbol in pairs_to_scan:
            if not bot_instance.user_prefs.can_send_signal(chat_id, symbol):
                continue
            
            try:
                signal, _ = await bot_instance.analyze_pair(symbol)
                
                if not signal or 'error' in signal:
                    continue
                
                if signal['confidence'] < Config.MINIMUM_CONFIDENCE:
                    continue
                
                # Format and send signal
                display_name = Config.CURRENCY_PAIRS[symbol]
                message = SignalFormatter.format_signal(signal, display_name)
                
                await context.bot.send_message(
                    chat_id=chat_id,
                    text=f"🤖 **AUTO-SIGNAL**\n\n{message}",
                    parse_mode='Markdown'
                )
                
                bot_instance.user_prefs.record_signal_sent(chat_id, symbol)
                print(f"✅ Auto-signal sent: {symbol} to {chat_id}")
                
                await asyncio.sleep(3)
                
            except Exception as e:
                print(f"❌ Error scanning {symbol}: {e}")
                continue

# ================================================
# BOT STARTUP FUNCTION (Call this from main())
# ================================================
async def start_bot():
    """Initialize and start the Telegram bot"""
    global bot_instance
    
    # Configure logging
    logging.basicConfig(
        format='%(asctime)s - %(name)s - %(levelname)s - %(message)s',
        level=logging.INFO
    )
    
    print("=" * 60)
    print("🚀 STREAMLINED 50/50 AI-BOT FOREX SIGNAL BOT")
    print("=" * 60)
    print(f"📊 Monitoring: {len(Config.CURRENCY_PAIRS)} pairs")
    print(f"💪 Min Confidence: {Config.MINIMUM_CONFIDENCE}%")
    print(f"⏱️ Scan Interval: {Config.AUTO_SCAN_INTERVAL} minutes")
    print(f"🤖 Mode: 50% Bot + 50% AI")
    print("=" * 60)
    
    # Initialize bot instance
    bot_instance = ForexSignalBot()
    
    # Create application
    application = Application.builder().token(Config.TELEGRAM_BOT_TOKEN).build()
    
    # Add command handlers
    application.add_handler(CommandHandler("start", start))
    application.add_handler(CommandHandler("analyze", analyze))
    application.add_handler(CommandHandler("pairs", pairs))
    application.add_handler(CommandHandler("status", status))
    application.add_handler(CommandHandler("auto_off", auto_off))
    
    # Add scheduled jobs
    job_queue = application.job_queue
    
    # Auto-scan markets
    job_queue.run_repeating(
        auto_scan_markets,
        interval=Config.AUTO_SCAN_INTERVAL * 60,
        first=60
    )
    
    print("\n✅ Bot is running!")
    print("📱 Users will receive automatic signals")
    print("⏸️ Press Ctrl+C to stop\n")
    
    # Start the bot
    await application.run_polling(allowed_updates=Update.ALL_TYPES)

def main():
    """Main function to start the bot - call this from your Flask wrapper"""
    asyncio.run(start_bot())

if __name__ == '__main__':
    main()