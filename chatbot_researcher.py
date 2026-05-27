import os
from dotenv import load_dotenv
load_dotenv()

# os.environ["OPENAI_API_KEY"] = os.getenv("OPENAI_API_KEY")
# os.environ["GOOGLE_API_KEY"] = os.getenv("GOOGLE_API_KEY")
os.environ["GROQ_API_KEY"] = os.getenv("GROQ_API_KEY")
os.environ['ALPHA_VINTAGE_API_KEY'] = os.getenv("ALPHA_VINTAGE_API_KEY")

from langchain_groq import ChatGroq

model = ChatGroq(
    model="llama-3.1-8b-instant",
    temperature=0
)

import requests

SYMBOL = input("Please provide the name of the Company or a Ticker: ")  

ALPHA_VINTAGE_API_KEY = os.getenv("ALPHA_VINTAGE_API_KEY")
url = f"https://www.alphavantage.co/query?function=TIME_SERIES_DAILY&symbol={SYMBOL}&apikey={ALPHA_VINTAGE_API_KEY}"

res = requests.get(url=url)
print(res.json())

import pandas as pd

data = res.json()['Time Series (Daily)']
df = pd.DataFrame(data).T  # Add .T to transpose

df.index.name = 'date'
df.columns = ['open', 'high', 'low', 'close', 'volume']

# Convert to appropriate data types
df = df.astype(float)

# Sort by date (newest first is default, or use ascending=True for oldest first)
df = df.sort_index(ascending=False)

print(df.head())
df.to_csv(f'time_series_{SYMBOL}.csv')

from ta.volatility import AverageTrueRange, BollingerBands, KeltnerChannel, DonchianChannel, UlcerIndex
from ta.trend import EMAIndicator, ADXIndicator, MACD, IchimokuIndicator, CCIIndicator, AroonIndicator, PSARIndicator
from ta.volume import VolumeWeightedAveragePrice, OnBalanceVolumeIndicator, ChaikinMoneyFlowIndicator, MFIIndicator, ForceIndexIndicator
from ta.momentum import RSIIndicator, UltimateOscillator, StochasticOscillator, WilliamsRIndicator, ROCIndicator, TSIIndicator, PercentagePriceOscillator
import numpy as np
import pandas as pd

def technical_analysis(df):
    # --- Volatility ---
    df['atr']          = AverageTrueRange(high=df['high'], low=df['low'], close=df['close'], window=10).average_true_range()
    res                = BollingerBands(close=df['close'], window=10, window_dev=2)
    df['bb_high']      = res.bollinger_hband()
    df['bb_low']       = res.bollinger_lband()
    df['bb_mid']       = res.bollinger_mavg()
    df['bb_width']     = res.bollinger_wband()
    df['bb_pct']       = res.bollinger_pband()
    kc                 = KeltnerChannel(high=df['high'], low=df['low'], close=df['close'], window=10)
    df['kc_high']      = kc.keltner_channel_hband()
    df['kc_low']       = kc.keltner_channel_lband()
    df['kc_mid']       = kc.keltner_channel_mband()
    dc                 = DonchianChannel(high=df['high'], low=df['low'], close=df['close'], window=10)
    df['dc_high']      = dc.donchian_channel_hband()
    df['dc_low']       = dc.donchian_channel_lband()
    df['dc_mid']       = dc.donchian_channel_mband()
    df['ulcer']        = UlcerIndex(close=df['close'], window=10).ulcer_index()

    # --- Trend ---
    df['ema_50']       = EMAIndicator(close=df['close'], window=50).ema_indicator()
    df['ema_20']       = EMAIndicator(close=df['close'], window=20).ema_indicator()
    df['ema_10']       = EMAIndicator(close=df['close'], window=10).ema_indicator()
    df['dema_50']      = EMAIndicator(close=df['ema_50'], window=50).ema_indicator()
    df['dema_20']      = EMAIndicator(close=df['ema_20'], window=20).ema_indicator()
    adx_ind            = ADXIndicator(high=df['high'], low=df['low'], close=df['close'], window=10)
    df['adx']          = adx_ind.adx()
    df['adx_pos']      = adx_ind.adx_pos()
    df['adx_neg']      = adx_ind.adx_neg()
    macd               = MACD(close=df['close'], window_slow=26, window_fast=12, window_sign=9)
    df['macd']         = macd.macd()
    df['macd_signal']  = macd.macd_signal()
    df['macd_diff']    = macd.macd_diff()
    df['cci']          = CCIIndicator(high=df['high'], low=df['low'], close=df['close'], window=10).cci()
    aroon              = AroonIndicator(high=df['high'], low=df['low'], window=10)
    df['aroon_up']     = aroon.aroon_up()
    df['aroon_down']   = aroon.aroon_down()
    df['aroon_ind']    = aroon.aroon_indicator()
    psar               = PSARIndicator(high=df['high'], low=df['low'], close=df['close'])
    df['psar']         = psar.psar()
    df['psar_up']      = psar.psar_up()
    df['psar_down']    = psar.psar_down()
    ichimoku           = IchimokuIndicator(high=df['high'], low=df['low'], window1=9, window2=26, window3=52)
    df['ichimoku_a']   = ichimoku.ichimoku_a()
    df['ichimoku_b']   = ichimoku.ichimoku_b()
    df['ichimoku_base']= ichimoku.ichimoku_base_line()
    df['ichimoku_conv']= ichimoku.ichimoku_conversion_line()

    # --- Volume ---
    df['vwap']         = VolumeWeightedAveragePrice(high=df['high'], low=df['low'], close=df['close'], volume=df['volume'], window=10).volume_weighted_average_price()
    df['obv']          = OnBalanceVolumeIndicator(close=df['close'], volume=df['volume']).on_balance_volume()
    df['cmf']          = ChaikinMoneyFlowIndicator(high=df['high'], low=df['low'], close=df['close'], volume=df['volume'], window=10).chaikin_money_flow()
    df['mfi']          = MFIIndicator(high=df['high'], low=df['low'], close=df['close'], volume=df['volume'], window=10).money_flow_index()
    df['fi']           = ForceIndexIndicator(close=df['close'], volume=df['volume'], window=10).force_index()

    # --- Momentum ---
    df['rsi']          = RSIIndicator(close=df['close'], window=10).rsi()
    df['uo']           = UltimateOscillator(high=df['high'], low=df['low'], close=df['close'], window1=7, window2=14, window3=21, weight1=4, weight2=2, weight3=1).ultimate_oscillator()
    stoch              = StochasticOscillator(high=df['high'], low=df['low'], close=df['close'], window=10, smooth_window=3)
    df['stoch']        = stoch.stoch()
    df['stoch_signal'] = stoch.stoch_signal()
    df['willr']        = WilliamsRIndicator(high=df['high'], low=df['low'], close=df['close'], lbp=10).williams_r()
    df['roc']          = ROCIndicator(close=df['close'], window=10).roc()
    tsi                = TSIIndicator(close=df['close'], window_slow=25, window_fast=13)
    df['tsi']          = tsi.tsi()
    ppo                = PercentagePriceOscillator(close=df['close'], window_slow=26, window_fast=12, window_sign=9)
    df['ppo']          = ppo.ppo()
    df['ppo_signal']   = ppo.ppo_signal()
    df['ppo_hist']     = ppo.ppo_hist()

    # Signal 1 : RSI — oversold / overbought
    df['col1']  = np.where(df['rsi'] < 30,  1, np.where(df['rsi'] > 70,  -1, 0))
    sig1        = df['col1'].iloc[-1]

    # Signal 2 : EMA-20 / EMA-50 crossover
    df['col2']  = np.where((df['ema_20'] > df['ema_50']) & (df['ema_20'].shift(1) <= df['ema_50'].shift(1)),  1,
                 np.where((df['ema_20'] < df['ema_50']) & (df['ema_20'].shift(1) >= df['ema_50'].shift(1)), -1, 0))
    sig2        = df['col2'].iloc[-1]

    # Signal 3 : Bollinger Band breakout — close above upper / below lower band
    df['col3']  = np.where(df['close'] > df['bb_high'],  1,
                 np.where(df['close'] < df['bb_low'],   -1, 0))
    sig3        = df['col3'].iloc[-1]

    # Signal 4 : MACD crossover
    df['col4']  = np.where((df['macd'] > df['macd_signal']) & (df['macd'].shift(1) <= df['macd_signal'].shift(1)),  1,
                 np.where((df['macd'] < df['macd_signal']) & (df['macd'].shift(1) >= df['macd_signal'].shift(1)), -1, 0))
    sig4        = df['col4'].iloc[-1]

    # Signal 5 : ADX — weak trend (<25) / strong trend (>40)
    df['col5']  = np.where(df['adx'] < 25,  1,
                 np.where(df['adx'] > 40,  -1, 0))   # -1 = strong trend (risk of continuation/exhaustion)
    sig5        = df['col5'].iloc[-1]

    # Signal 6 : VWAP spike — above 1.5× mean (bullish surge) / below 0.5× mean (bearish collapse)
    vwap_mean   = df['vwap'].rolling(10).mean()
    df['col6']  = np.where(df['vwap'] > vwap_mean * 1.5,  1,
                 np.where(df['vwap'] < vwap_mean * 0.5, -1, 0))
    sig6        = df['col6'].iloc[-1]

    # Signal 7 : OBV — extreme accumulation (>2× mean) / extreme distribution (<0.5× mean)
    obv_mean    = df['obv'].rolling(10).mean()
    df['col7']  = np.where(df['obv'] > obv_mean * 2,    1,
                 np.where(df['obv'] < obv_mean * 0.5, -1, 0))
    sig7        = df['col7'].iloc[-1]

    # Signal 8 : Stochastic %K — oversold / overbought
    df['col8']  = np.where(df['stoch'] < 30,  1, np.where(df['stoch'] > 70,  -1, 0))
    sig8        = df['col8'].iloc[-1]

    # Signal 9 : Ultimate Oscillator — bullish (>50) / bearish (<30)
    df['col9']  = np.where(df['uo'] > 50,  1, np.where(df['uo'] < 30,  -1, 0))
    sig9        = df['col9'].iloc[-1]

    # Signal 10 : CMF — buying pressure (>0.1) / selling pressure (<-0.1)
    df['col10'] = np.where(df['cmf'] > 0.1,  1, np.where(df['cmf'] < -0.1, -1, 0))
    sig10       = df['col10'].iloc[-1]

    # Signal 11 : MFI — oversold / overbought
    df['col11'] = np.where(df['mfi'] < 20,  1, np.where(df['mfi'] > 80,  -1, 0))
    sig11       = df['col11'].iloc[-1]

    # Signal 12 : CCI — oversold / overbought
    df['col12'] = np.where(df['cci'] < -100,  1, np.where(df['cci'] > 100,  -1, 0))
    sig12       = df['col12'].iloc[-1]

    # Signal 13 : Aroon crossover — Up crosses above Down / Down crosses above Up
    df['col13'] = np.where((df['aroon_up'] > df['aroon_down']) & (df['aroon_up'].shift(1) <= df['aroon_down'].shift(1)),  1,
                 np.where((df['aroon_up'] < df['aroon_down']) & (df['aroon_up'].shift(1) >= df['aroon_down'].shift(1)), -1, 0))
    sig13       = df['col13'].iloc[-1]

    # Signal 14 : Williams %R — oversold / overbought
    df['col14'] = np.where(df['willr'] < -80,  1, np.where(df['willr'] > -20,  -1, 0))
    sig14       = df['col14'].iloc[-1]

    # Signal 15 : TSI — crosses above zero (bullish) / crosses below zero (bearish)
    df['col15'] = np.where((df['tsi'] > 0) & (df['tsi'].shift(1) <= 0),  1,
                 np.where((df['tsi'] < 0) & (df['tsi'].shift(1) >= 0), -1, 0))
    sig15       = df['col15'].iloc[-1]

    signals = {
        'sig1_rsi':           sig1,
        'sig2_ema_cross':     sig2,
        'sig3_bb_break':      sig3,
        'sig4_macd_cross':    sig4,
        'sig5_adx':           sig5,
        'sig6_vwap_spike':    sig6,
        'sig7_obv':           sig7,
        'sig8_stoch':         sig8,
        'sig9_uo':            sig9,
        'sig10_cmf':          sig10,
        'sig11_mfi':          sig11,
        'sig12_cci':          sig12,
        'sig13_aroon_cross':  sig13,
        'sig14_willr':        sig14,
        'sig15_tsi_cross':    sig15,
    }

    technical_params = {
        # Volatility
        'atr':           df['atr'].iloc[-1],
        'bb_high':       df['bb_high'].iloc[-1],
        'bb_low':        df['bb_low'].iloc[-1],
        'bb_mid':        df['bb_mid'].iloc[-1],
        'bb_width':      df['bb_width'].iloc[-1],
        'bb_pct':        df['bb_pct'].iloc[-1],
        'kc_high':       df['kc_high'].iloc[-1],
        'kc_low':        df['kc_low'].iloc[-1],
        'kc_mid':        df['kc_mid'].iloc[-1],
        'dc_high':       df['dc_high'].iloc[-1],
        'dc_low':        df['dc_low'].iloc[-1],
        'dc_mid':        df['dc_mid'].iloc[-1],
        'ulcer':         df['ulcer'].iloc[-1],
        # Trend
        'ema_10':        df['ema_10'].iloc[-1],
        'ema_20':        df['ema_20'].iloc[-1],
        'ema_50':        df['ema_50'].iloc[-1],
        'dema_20':       df['dema_20'].iloc[-1],
        'dema_50':       df['dema_50'].iloc[-1],
        'adx':           df['adx'].iloc[-1],
        'adx_pos':       df['adx_pos'].iloc[-1],
        'adx_neg':       df['adx_neg'].iloc[-1],
        'macd':          df['macd'].iloc[-1],
        'macd_signal':   df['macd_signal'].iloc[-1],
        'macd_diff':     df['macd_diff'].iloc[-1],
        'cci':           df['cci'].iloc[-1],
        'aroon_up':      df['aroon_up'].iloc[-1],
        'aroon_down':    df['aroon_down'].iloc[-1],
        'aroon_ind':     df['aroon_ind'].iloc[-1],
        'psar':          df['psar'].iloc[-1],
        'psar_up':       df['psar_up'].iloc[-1],
        'psar_down':     df['psar_down'].iloc[-1],
        'ichimoku_a':    df['ichimoku_a'].iloc[-1],
        'ichimoku_b':    df['ichimoku_b'].iloc[-1],
        'ichimoku_base': df['ichimoku_base'].iloc[-1],
        'ichimoku_conv': df['ichimoku_conv'].iloc[-1],
        # Volume
        'vwap':          df['vwap'].iloc[-1],
        'obv':           df['obv'].iloc[-1],
        'cmf':           df['cmf'].iloc[-1],
        'mfi':           df['mfi'].iloc[-1],
        'fi':            df['fi'].iloc[-1],
        # Momentum
        'rsi':           df['rsi'].iloc[-1],
        'uo':            df['uo'].iloc[-1],
        'stoch':         df['stoch'].iloc[-1],
        'stoch_signal':  df['stoch_signal'].iloc[-1],
        'willr':         df['willr'].iloc[-1],
        'roc':           df['roc'].iloc[-1],
        'tsi':           df['tsi'].iloc[-1],
        'ppo':           df['ppo'].iloc[-1],
        'ppo_signal':    df['ppo_signal'].iloc[-1],
        'ppo_hist':      df['ppo_hist'].iloc[-1],
    }

    return technical_params, signals


df = pd.read_csv(f'time_series_{SYMBOL}.csv')
technical_params, signals = technical_analysis(df)

pd.DataFrame([technical_params]).to_csv(f'technical_analysis_{SYMBOL}.csv')
pd.DataFrame([signals]).to_csv(f'signals_{SYMBOL}.csv')

technical_params, signals

"""
Screener.in Web Scraper for Indian Stock Fundamental Analysis
Extracts key fundamental data including ratios, financials, and company info
"""

import requests
from bs4 import BeautifulSoup
import pandas as pd
import time
from typing import Dict, Optional
import re


class ScreenerScraper:
    """Scraper for Screener.in fundamental data"""
    
    def __init__(self):
        self.base_url = "https://www.screener.in"
        self.session = requests.Session()
        self.session.headers.update({
            'User-Agent': 'Mozilla/5.0 (Windows NT 10.0; Win64; x64) AppleWebKit/537.36 (KHTML, like Gecko) Chrome/120.0.0.0 Safari/537.36'
        })
    
    def get_stock_data(self, symbol: str) -> Optional[Dict]:
        """
        Fetch fundamental data for a stock
        
        Args:
            symbol: Stock symbol (e.g., 'RELIANCE', 'TCS', 'INFY')
        
        Returns:
            Dictionary containing fundamental data
        """
        url = f"{self.base_url}/company/{symbol}/"
        
        try:
            response = self.session.get(url, timeout=10)
            response.raise_for_status()
            
            soup = BeautifulSoup(response.content, 'html.parser')
            
            # Extract all fundamental data
            data = {
                'symbol': symbol,
                'company_name': self._get_company_name(soup),
                'current_price': self._get_current_price(soup),
                'market_cap': self._get_market_cap(soup),
                **self._get_key_ratios(soup),
                **self._get_quarterly_results(soup),
                **self._get_profit_loss(soup),
                **self._get_balance_sheet(soup),
                **self._get_cash_flow(soup),
                **self._get_shareholding_pattern(soup)
            }
            
            return data
            
        except requests.exceptions.RequestException as e:
            print(f"Error fetching data for {symbol}: {e}")
            return None
    
    def _get_company_name(self, soup: BeautifulSoup) -> str:
        """Extract company name"""
        try:
            name_tag = soup.find('h1', class_='h2')
            if name_tag:
                return name_tag.get_text(strip=True)
        except:
            pass
        return "N/A"
    
    def _get_current_price(self, soup: BeautifulSoup) -> float:
        """Extract current stock price"""
        try:
            price_tag = soup.find('span', class_='number')
            if price_tag:
                price_text = price_tag.get_text(strip=True).replace(',', '')
                return float(price_text)
        except:
            pass
        return None
    
    def _get_market_cap(self, soup: BeautifulSoup) -> str:
        """Extract market capitalization"""
        try:
            # Find the market cap in the ratios section
            ratios = soup.find_all('li', class_='flex flex-space-between')
            for ratio in ratios:
                name = ratio.find('span', class_='name')
                if name and 'Market Cap' in name.get_text():
                    value = ratio.find('span', class_='number')
                    if value:
                        return value.get_text(strip=True)
        except:
            pass
        return "N/A"
    
    def _get_key_ratios(self, soup: BeautifulSoup) -> Dict:
        """Extract key financial ratios"""
        ratios = {}
        
        ratio_names = {
            'Market Cap': 'market_cap',
            'Current Price': 'current_price_ratio',
            'High / Low': 'high_low',
            'Stock P/E': 'pe_ratio',
            'Book Value': 'book_value',
            'Dividend Yield': 'dividend_yield',
            'ROCE': 'roce',
            'ROE': 'roe',
            'Face Value': 'face_value',
            'EPS': 'eps',
            'P/B': 'pb_ratio',
            'Debt to Equity': 'debt_to_equity'
        }
        
        try:
            ratio_elements = soup.find_all('li', class_='flex flex-space-between')
            
            for element in ratio_elements:
                name_tag = element.find('span', class_='name')
                value_tag = element.find('span', class_='number')
                
                if name_tag and value_tag:
                    name = name_tag.get_text(strip=True)
                    value = value_tag.get_text(strip=True)
                    
                    # Map to our standardized names
                    for key, standard_name in ratio_names.items():
                        if key in name:
                            ratios[standard_name] = value
                            break
        except Exception as e:
            print(f"Error extracting ratios: {e}")
        
        return ratios
    
    def _get_quarterly_results(self, soup: BeautifulSoup) -> Dict:
        """Extract latest quarterly results"""
        quarterly_data = {}
        
        try:
            # Find quarterly results table
            tables = soup.find_all('table', class_='data-table')
            
            for table in tables:
                header = table.find_previous('h2')
                if header and 'Quarterly Results' in header.get_text():
                    # Get the latest quarter (first data column)
                    rows = table.find_all('tr')
                    
                    for row in rows:
                        cells = row.find_all(['th', 'td'])
                        if len(cells) >= 2:
                            metric = cells[0].get_text(strip=True)
                            latest_value = cells[1].get_text(strip=True)
                            
                            # Map common metrics
                            metric_map = {
                                'Sales': 'quarterly_sales',
                                'Operating Profit': 'quarterly_operating_profit',
                                'Net Profit': 'quarterly_net_profit',
                                'EPS in Rs': 'quarterly_eps'
                            }
                            
                            for key, value in metric_map.items():
                                if key in metric:
                                    quarterly_data[value] = latest_value
                                    break
        except Exception as e:
            print(f"Error extracting quarterly results: {e}")
        
        return quarterly_data
    
    def _get_profit_loss(self, soup: BeautifulSoup) -> Dict:
        """Extract annual profit & loss data"""
        pl_data = {}
        
        try:
            tables = soup.find_all('table', class_='data-table')
            
            for table in tables:
                header = table.find_previous('h2')
                if header and 'Profit & Loss' in header.get_text():
                    rows = table.find_all('tr')
                    
                    for row in rows:
                        cells = row.find_all(['th', 'td'])
                        if len(cells) >= 2:
                            metric = cells[0].get_text(strip=True)
                            latest_value = cells[-1].get_text(strip=True)  # Get most recent year
                            
                            metric_map = {
                                'Sales': 'annual_sales',
                                'Operating Profit': 'annual_operating_profit',
                                'Net Profit': 'annual_net_profit',
                                'EPS in Rs': 'annual_eps'
                            }
                            
                            for key, value in metric_map.items():
                                if key in metric:
                                    pl_data[value] = latest_value
                                    break
        except Exception as e:
            print(f"Error extracting P&L: {e}")
        
        return pl_data
    
    def _get_balance_sheet(self, soup: BeautifulSoup) -> Dict:
        """Extract balance sheet data"""
        bs_data = {}
        
        try:
            tables = soup.find_all('table', class_='data-table')
            
            for table in tables:
                header = table.find_previous('h2')
                if header and 'Balance Sheet' in header.get_text():
                    rows = table.find_all('tr')
                    
                    for row in rows:
                        cells = row.find_all(['th', 'td'])
                        if len(cells) >= 2:
                            metric = cells[0].get_text(strip=True)
                            latest_value = cells[-1].get_text(strip=True)
                            
                            metric_map = {
                                'Total Assets': 'total_assets',
                                'Total Liabilities': 'total_liabilities',
                                'Equity Capital': 'equity_capital',
                                'Reserves': 'reserves'
                            }
                            
                            for key, value in metric_map.items():
                                if key in metric:
                                    bs_data[value] = latest_value
                                    break
        except Exception as e:
            print(f"Error extracting balance sheet: {e}")
        
        return bs_data
    
    def _get_cash_flow(self, soup: BeautifulSoup) -> Dict:
        """Extract cash flow data"""
        cf_data = {}
        
        try:
            tables = soup.find_all('table', class_='data-table')
            
            for table in tables:
                header = table.find_previous('h2')
                if header and 'Cash Flow' in header.get_text():
                    rows = table.find_all('tr')
                    
                    for row in rows:
                        cells = row.find_all(['th', 'td'])
                        if len(cells) >= 2:
                            metric = cells[0].get_text(strip=True)
                            latest_value = cells[-1].get_text(strip=True)
                            
                            metric_map = {
                                'Operating Activities': 'operating_cash_flow',
                                'Investing Activities': 'investing_cash_flow',
                                'Financing Activities': 'financing_cash_flow'
                            }
                            
                            for key, value in metric_map.items():
                                if key in metric:
                                    cf_data[value] = latest_value
                                    break
        except Exception as e:
            print(f"Error extracting cash flow: {e}")
        
        return cf_data
    
    def _get_shareholding_pattern(self, soup: BeautifulSoup) -> Dict:
        """Extract shareholding pattern"""
        shareholding = {}
        
        try:
            # Find shareholding section
            sections = soup.find_all('section')
            
            for section in sections:
                header = section.find('h2')
                if header and 'Shareholding Pattern' in header.get_text():
                    # Extract promoter and FII/DII holdings
                    text = section.get_text()
                    
                    # Use regex to find percentages
                    promoter_match = re.search(r'Promoters[:\s]+(\d+\.?\d*)%?', text)
                    fii_match = re.search(r'FIIs[:\s]+(\d+\.?\d*)%?', text)
                    dii_match = re.search(r'DIIs[:\s]+(\d+\.?\d*)%?', text)
                    
                    if promoter_match:
                        shareholding['promoter_holding'] = f"{promoter_match.group(1)}%"
                    if fii_match:
                        shareholding['fii_holding'] = f"{fii_match.group(1)}%"
                    if dii_match:
                        shareholding['dii_holding'] = f"{dii_match.group(1)}%"
        except Exception as e:
            print(f"Error extracting shareholding: {e}")
        
        return shareholding
    
    def scrape_multiple_stocks(self, symbols: list, delay: float = 2.0) -> pd.DataFrame:
        """
        Scrape data for multiple stocks
        
        Args:
            symbols: List of stock symbols
            delay: Delay between requests in seconds (be respectful!)
        
        Returns:
            DataFrame with all stock data
        """
        all_data = []
        
        for i, symbol in enumerate(symbols):
            print(f"Scraping {symbol} ({i+1}/{len(symbols)})...")
            
            data = self.get_stock_data(symbol)
            if data:
                all_data.append(data)
            
            # Be respectful - add delay between requests
            if i < len(symbols) - 1:
                time.sleep(delay)
        
        df = pd.DataFrame(all_data)
        return df

import requests
import json
from datetime import datetime, timedelta


def get_news(api_key, company_name, num_articles=100, days_ago=10):
    start_date = (datetime.now() - timedelta(days=days_ago)).strftime('%Y-%m-%d')

    endpoint = "https://newsapi.org/v2/everything"

    params = {
        'apiKey': api_key,
        'q': company_name,
        'sortBy': 'publishedAt',
        'language': 'en',
        'pageSize': num_articles,
        'from': start_date
    }

    try:
        response = requests.get(endpoint, params=params)

        if response.status_code == 200:
            data = response.json()

            articles = data.get('articles', [])

            filtered_articles = []

            for article in articles:
                title = article.get('title', '')
                description = article.get('description', '')
                url = article.get('url', '')

                if company_name.lower() in title.lower():
                    print(f"Title: {title}")
                    print(f"Description: {description}")
                    print(f"URL: {url}")
                    print("\n" + "="*50 + "\n")

                    filtered_articles.append({
                        "title": title,
                        "description": description,
                        "url": url
                    })

            return filtered_articles   # RETURN DATA HERE

        else:
            print(f"Error: {response.status_code} - {response.text}")
            return None

    except Exception as e:
        print(f"An error occurred: {e}")
        return None

api_key = os.getenv('NEWS_TOOL_API_KEY')

res = get_news(api_key, SYMBOL, num_articles=100, days_ago=10)

if res:
    with open("response.txt", "w", encoding="utf-8") as f:
        json.dump(res, f, indent=4)

    print("Saved to response.txt")
else:
    print("No data saved.")

import requests
import os

def get_fear_greed_index():

    api_key = os.getenv("CNC_TOOL_API_KEY")

    url = "https://pro-api.coinmarketcap.com/v3/fear-and-greed/latest"

    headers = {
        "Accept": "application/json",
        "X-CMC_PRO_API_KEY": api_key
    }

    try:
        response = requests.get(url, headers=headers, timeout=10)

        data = response.json()["data"]

        return {
            "score": data["value"],
            "classification": data["value_classification"],
            "last_update": data["update_time"]
        }

    except Exception as e:
        return {
            "error": str(e)
        }

from langchain.tools import tool
from langchain_core.prompts import ChatPromptTemplate
from langchain_groq import ChatGroq
from langgraph.prebuilt import create_react_agent
import pandas as pd

@tool
def get_news_tool(symbol: str) -> str:
    """Fetch latest news about a company."""
    api_key = os.getenv('NEWS_TOOL_API_KEY')
    return str(get_news(api_key, SYMBOL))

@tool
def technical_analysis_tool(symbol: str) -> str:
    """Run technical analysis on a stock and return technical params and signals."""
    df = pd.read_csv(f'time_series_{SYMBOL}.csv')
    return str(technical_analysis(df))

@tool
def screener_data_tool(symbol: str) -> str:
    """Fetch fundamental stock data from Screener."""
    symbol = SYMBOL
    scraper = ScreenerScraper()
    return str(scraper.get_stock_data(symbol))

@tool
def calculate_tp_sl(close: float, atr: float, signal: int) -> dict:
    """
    Calculate Take Profit and Stop Loss using close price, ATR and signal direction.
    BUY  (signal =  1): TP = close + 2*ATR | SL = close - 1*ATR
    SELL (signal = -1): TP = close - 2*ATR | SL = close + 1*ATR
    """
    if signal == 1:   # BUY
        return {
            "recommendation" : "BUY",
            "take_profit"    : round(close + 2 * atr, 2),  # higher than close
            "stop_loss"      : round(close - 1 * atr, 2)   # lower than close
        }
    elif signal == -1:  # SELL
        return {
            "recommendation" : "SELL",
            "take_profit"    : round(close - 2 * atr, 2),  # lower than close
            "stop_loss"      : round(close + 1 * atr, 2)   # higher than close
        }
    else:               # NEUTRAL
        return {
            "recommendation" : "HOLD",
            "take_profit"    : None,
            "stop_loss"      : None
        }

@tool
def hold_signal() -> dict:
    """
    Use this tool ONLY when signal = 0 (NEUTRAL).
    No TP or SL should be calculated for neutral signals.
    """
    return {
        "recommendation" : "HOLD",
        "take_profit"    : "N/A — signal is neutral, no trade recommended",
        "stop_loss"      : "N/A — signal is neutral, no trade recommended"
    }

@tool
def fear_greed_tool() -> str:
    """
    Fetch the crypto market Fear & Greed Index from CoinMarketCap.
    Useful for market sentiment and regime analysis.
    """

    return str(get_fear_greed_index())

# model = ChatGroq(
#     model="llama-3.1-8b-instant",
#     temperature=0
# )

model = ChatGroq(
    model="llama-3.3-70b-versatile",  # much better instruction following
    temperature=0
)

System_message = """You are an expert financial assistant specializing in stock market analysis.
You have access to five tools: technical_analysis_tool, screener_data_tool, get_news_tool,
calculate_tp_sl_buy, calculate_tp_sl_sell, and hold_signal.

═══════════════════════════════════════════════
STEP 1 — ANALYSIS MODE SELECTION
═══════════════════════════════════════════════

When the user first provides their request, ask them to choose an analysis mode:

  1 → Technical Analysis
  2 → Fundamental Analysis
  3 → News Analysis

Wait for the user's choice before doing anything else.

═══════════════════════════════════════════════
STEP 2 — IF USER CHOSE "1" (TECHNICAL)
═══════════════════════════════════════════════

After the user selects "1", DO NOT run any tool yet.
Ask the user to select a specific signal strategy (1-15):

  Signal 1  → RSI oversold/overbought        (< 30 = bullish | > 70 = bearish)
  Signal 2  → EMA-20/50 crossover            (cross up = bullish | cross down = bearish)
  Signal 3  → Bollinger Band breakout        (above upper = bullish | below lower = bearish)
  Signal 4  → MACD crossover                 (above signal line = bullish | below = bearish)
  Signal 5  → ADX trend strength             (< 25 = weak trend | > 40 = strong trend)
  Signal 6  → VWAP spike                     (> 1.5x mean = bullish | < 0.5x mean = bearish)
  Signal 7  → OBV accumulation               (> 2x mean = bullish | < 0.5x mean = bearish)
  Signal 8  → Stochastic %K                  (< 30 = bullish | > 70 = bearish)
  Signal 9  → Ultimate Oscillator            (> 50 = bullish | < 30 = bearish)
  Signal 10 → CMF Chaikin Money Flow         (> 0.1 = bullish | < -0.1 = bearish)
  Signal 11 → MFI Money Flow Index           (< 20 = bullish | > 80 = bearish)
  Signal 12 → CCI Commodity Channel Index    (< -100 = bullish | > 100 = bearish)
  Signal 13 → Aroon crossover                (Up > Down = bullish | Down > Up = bearish)
  Signal 14 → Williams %R                    (< -80 = bullish | > -20 = bearish)
  Signal 15 → TSI zero cross                 (cross above 0 = bullish | cross below 0 = bearish)

Wait for the user to pick a signal number before proceeding.

═══════════════════════════════════════════════
STEP 3 — EXECUTE TECHNICAL ANALYSIS
═══════════════════════════════════════════════

Once the user has selected a signal number (1-15), follow these steps IN ORDER:

  STEP 3A — Call technical_analysis_tool with the stock symbol.

  STEP 3B — Read the signal value for the chosen signal number from the output:
               sig_value = 1  → BULLISH
               sig_value = -1 → BEARISH
               sig_value = 0  → NEUTRAL

  STEP 3C — Based on sig_value, call EXACTLY ONE of these three tools:

    ┌─────────────────────────────────────────────────────────────┐
    │ sig_value =  1 → call calculate_tp_sl_buy(close, atr)      │
    │                  TP = close + 2xATR  (ABOVE close price)   │
    │                  SL = close - 2xATR  (BELOW close price)   │
    ├─────────────────────────────────────────────────────────────┤
    │ sig_value = -1 → call calculate_tp_sl_sell(close, atr)     │
    │                  TP = close - 2xATR  (BELOW close price)   │
    │                  SL = close + 2xATR  (ABOVE close price)   │
    ├─────────────────────────────────────────────────────────────┤
    │ sig_value =  0 → call hold_signal()                        │
    │                  NO TP or SL — do not calculate anything   │
    └─────────────────────────────────────────────────────────────┘

  STEP 3D — Present the final output in this exact format:

    ─────────────────────────────────────────
    Stock        : <SYMBOL>
    Strategy     : Signal <N> — <Signal Name>
    Signal Value : <1 / -1 / 0>
    Direction    : <BULLISH / BEARISH / NEUTRAL>
    ─────────────────────────────────────────
    Recommendation : BUY  /  SELL  /  HOLD
    Close Price    : <value>
    ATR            : <value>
    Take Profit    : <TP value>  or  N/A (if HOLD)
    Stop Loss      : <SL value>  or  N/A (if HOLD)
    ─────────────────────────────────────────
    Justification  : <2-3 sentences using the chosen signal
                      and 1-2 supporting indicators>

═══════════════════════════════════════════════
CRITICAL RULES FOR STEP 3 — NEVER VIOLATE
═══════════════════════════════════════════════

  ✗ NEVER compute TP/SL manually — always use the tools
  ✗ NEVER call calculate_tp_sl_buy or calculate_tp_sl_sell when sig_value = 0
  ✗ NEVER call hold_signal when sig_value = 1 or -1
  ✗ NEVER show TP above close for a SELL signal
  ✗ NEVER show TP below close for a BUY signal
  ✗ NEVER show SL above close for a BUY signal
  ✗ NEVER show SL below close for a SELL signal
  ✗ NEVER skip STEP 3C — always call one of the three tools

═══════════════════════════════════════════════
IF USER CHOSE "2" (FUNDAMENTAL)
═══════════════════════════════════════════════

  1. Call screener_data_tool with the stock symbol
  2. Analyze key valuation metrics:
       P/E, P/B, ROE, ROCE, Debt/Equity, EPS growth
  3. Present output in this format:

    ─────────────────────────────────────────
    Stock          : <SYMBOL>
    Mode           : Fundamental Analysis
    ─────────────────────────────────────────
    Recommendation : BUY  /  SELL  /  HOLD
    ─────────────────────────────────────────
    P/E            : <value>
    ROE            : <value>
    ROCE           : <value>
    Debt/Equity    : <value>
    EPS            : <value>
    ─────────────────────────────────────────
    Justification  : <2-3 sentence fundamental case>

═══════════════════════════════════════════════
IF USER CHOSE "3" (NEWS)
═══════════════════════════════════════════════

  1. Call get_news_tool with the stock symbol
  2. Determine overall sentiment from headlines:
       Mostly positive → BULLISH → BUY
       Mostly negative → BEARISH → SELL
       Mixed / unclear → NEUTRAL → HOLD
  3. Present output in this format:

    ─────────────────────────────────────────
    Stock          : <SYMBOL>
    Mode           : News Analysis
    Sentiment      : <BULLISH / BEARISH / NEUTRAL>
    ─────────────────────────────────────────
    Recommendation : BUY  /  SELL  /  HOLD
    ─────────────────────────────────────────
    Key Headlines:
      • <headline 1>
      • <headline 2>
      • <headline 3>
    ─────────────────────────────────────────
    Justification  : <2-3 sentence news-driven case>

═══════════════════════════════════════════════
GENERAL RULES
═══════════════════════════════════════════════

  - Always extract the stock symbol from the user's message before calling any tool
  - If no stock symbol is found, ask the user before proceeding
  - Never skip the two-step flow for technical mode (Step 1 → Step 2 → Step 3)
  - Keep all responses concise, structured, and actionable
"""

tools = [technical_analysis_tool, get_news_tool, screener_data_tool, calculate_tp_sl, hold_signal, fear_greed_tool]

agent_executor = create_react_agent(
    model=model,
    tools=tools,
    prompt=System_message
)

strategy = 1
response = agent_executor.invoke({
    "messages": [
        {"role": "user", "content": f"Give me technical analysis of {SYMBOL} using strategy {strategy}"}
    ]
})

print(response['messages'][-1].content)