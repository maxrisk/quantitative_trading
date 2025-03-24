#!/usr/bin/env python
# -*- coding: utf-8 -*-

import os
import time
import pandas as pd
import numpy as np
import requests
from datetime import datetime, timedelta
from dotenv import load_dotenv
from longport.openapi import Config, TradeContext, QuoteContext, Period, AdjustType, SecurityQuote, Candlestick, TradeStatus, TradeSession, MarketTradingSession, PrePostQuote, Brokers, ParticipantInfo, IntradayLine

# 加载环境变量
load_dotenv()

# 长桥API配置
app_key = os.getenv('LONGPORT_APP_KEY')
app_secret = os.getenv('LONGPORT_APP_SECRET')
access_token = os.getenv('LONGPORT_ACCESS_TOKEN')
deepseek_api_key = os.getenv('DEEPSEEK_API_KEY')

config = Config(app_key=app_key, app_secret=app_secret, access_token=access_token)

# DeepSeek API调用函数
def ask_deepseek(prompt):
    """使用DeepSeek API进行分析"""
    url = "https://api.deepseek.com/v1/chat/completions"
    headers = {
        "Content-Type": "application/json",
        "Authorization": f"Bearer {deepseek_api_key}"
    }
    data = {
        "model": "deepseek-chat",
        "messages": [{"role": "user", "content": prompt}],
        "temperature": 0.7,
        "max_tokens": 800
    }
    
    response = requests.post(url, headers=headers, json=data)
    if response.status_code == 200:
        return response.json()["choices"][0]["message"]["content"]
    else:
        return f"API调用失败: {response.status_code}, {response.text}"

# 初始化行情上下文
async def init_quote_context():
    quote_ctx = QuoteContext(config)
    return quote_ctx

# 行情回调函数
async def on_quote(symbol: str, quote: SecurityQuote):
    print(f"收到行情更新: {symbol}")
    print(f"最新价: {quote.last_done}")
    print(f"涨跌幅: {quote.change_rate}%")

# 获取K线数据
async def get_candlesticks(quote_ctx, symbol, period=Period.Day, count=30):
    candlesticks = quote_ctx.candlesticks(symbol=symbol, period=period, count=count, adjust_type=AdjustType.ForwardAdjust)
    
    # 转换为pandas DataFrame
    data = []
    for candle in candlesticks:
        data.append({
            'time': candle.timestamp,
            'open': candle.open,
            'high': candle.high,
            'low': candle.low,
            'close': candle.close,
            'volume': candle.volume,
            'turnover': candle.turnover
        })
    
    df = pd.DataFrame(data)
    return df

# 计算技术指标
def calculate_indicators(df):
    # 计算移动平均线
    df['MA5'] = df['close'].rolling(window=5).mean()
    df['MA10'] = df['close'].rolling(window=10).mean()
    df['MA20'] = df['close'].rolling(window=20).mean()
    
    # 计算相对强弱指标 (RSI)
    delta = df['close'].diff()
    gain = delta.where(delta > 0, 0)
    loss = -delta.where(delta < 0, 0)
    avg_gain = gain.rolling(window=14).mean()
    avg_loss = loss.rolling(window=14).mean()
    rs = avg_gain / avg_loss
    df['RSI'] = 100 - (100 / (1 + rs))
    
    # 计算MACD
    df['EMA12'] = df['close'].ewm(span=12, adjust=False).mean()
    df['EMA26'] = df['close'].ewm(span=26, adjust=False).mean()
    df['MACD'] = df['EMA12'] - df['EMA26']
    df['Signal'] = df['MACD'].ewm(span=9, adjust=False).mean()
    df['Histogram'] = df['MACD'] - df['Signal']
    
    # 布林带
    df['MA20'] = df['close'].rolling(window=20).mean()
    df['Upper_Band'] = df['MA20'] + (df['close'].rolling(window=20).std() * 2)
    df['Lower_Band'] = df['MA20'] - (df['close'].rolling(window=20).std() * 2)
    
    return df

# 生成交易信号
def generate_signals(df):
    df['Signal'] = 0  # 0: 持有, 1: 买入, -1: 卖出
    
    # 黄金交叉和死亡交叉
    df.loc[(df['MA5'] > df['MA10']) & (df['MA5'].shift(1) <= df['MA10'].shift(1)), 'Signal'] = 1  # 黄金交叉
    df.loc[(df['MA5'] < df['MA10']) & (df['MA5'].shift(1) >= df['MA10'].shift(1)), 'Signal'] = -1  # 死亡交叉
    
    # RSI超买超卖
    df.loc[df['RSI'] < 30, 'Signal'] = 1  # RSI低于30，超卖信号
    df.loc[df['RSI'] > 70, 'Signal'] = -1  # RSI高于70，超买信号
    
    # MACD信号
    df.loc[(df['MACD'] > df['Signal']) & (df['MACD'].shift(1) <= df['Signal'].shift(1)), 'Signal'] = 1
    df.loc[(df['MACD'] < df['Signal']) & (df['MACD'].shift(1) >= df['Signal'].shift(1)), 'Signal'] = -1
    
    return df

# 回测策略
def backtest(df, initial_capital=100000):
    df['Position'] = 0
    df['Cash'] = initial_capital
    df['Holdings'] = 0
    df['Portfolio'] = initial_capital
    
    position = 0
    cash = initial_capital
    
    for i in range(1, len(df)):
        # 前一天的信号，今天执行
        signal = df.iloc[i-1]['Signal']
        price = df.iloc[i]['open']  # 使用开盘价
        
        if signal == 1 and position == 0:  # 买入信号
            shares_to_buy = int(cash / price)
            holdings = shares_to_buy * price
            cash -= holdings
            position = shares_to_buy
        elif signal == -1 and position > 0:  # 卖出信号
            cash += position * price
            position = 0
            holdings = 0
        else:  # 持有
            holdings = position * price
            
        df.at[i, 'Position'] = position
        df.at[i, 'Cash'] = float(cash)
        df.at[i, 'Holdings'] = float(holdings)
        df.at[i, 'Portfolio'] = float(cash + holdings)
    
    # 计算回测结果
    df['Returns'] = df['Portfolio'].pct_change()
    cumulative_returns = float((df.iloc[-1]['Portfolio'] / initial_capital - 1) * 100)
    annualized_returns = float(((1 + cumulative_returns/100) ** (252/len(df)) - 1) * 100)
    
    return df, cumulative_returns, annualized_returns

# 智能分析市场
def analyze_market_with_ai(df):
    # 获取最近的数据
    recent_data = df.tail(10).to_dict(orient='records')
    data_str = "\n".join([f"日期: {row['time']}, 开盘: {row['open']}, 收盘: {row['close']}, 最高: {row['high']}, 最低: {row['low']}, 成交量: {row['volume']}" for row in recent_data])
    
    prompt = f"""
    我需要对以下股票数据进行专业的量化分析，请提供专业的市场分析和未来可能的趋势预测：
    
    {data_str}
    
    以下是一些技术指标：
    RSI: {df['RSI'].iloc[-1]:.2f}
    MACD: {df['MACD'].iloc[-1]:.2f}
    信号线: {df['Signal'].iloc[-1]:.2f}
    MA5: {df['MA5'].iloc[-1]:.2f}
    MA10: {df['MA10'].iloc[-1]:.2f}
    MA20: {df['MA20'].iloc[-1]:.2f}
    
    请提供专业的技术分析和未来一周可能的价格走势预测。
    """
    
    analysis = ask_deepseek(prompt)
    return analysis

# 主函数
async def main():
    """主函数"""
    try:
        # 初始化行情上下文
        quote_ctx = await init_quote_context()
        
        # 解析命令行参数
        import sys
        symbol = "700.HK"  # 默认为腾讯控股
        
        # 如果提供了命令行参数，使用提供的股票代码
        if len(sys.argv) > 1:
            symbol = sys.argv[1]
        
        print(f"正在获取 {symbol} 的K线数据...")
        df = await get_candlesticks(quote_ctx, symbol, Period.Day, 100)
        
        print("计算技术指标...")
        df = calculate_indicators(df)
        
        print("生成交易信号...")
        df = generate_signals(df)
        
        print("进行回测...")
        backtest_results, cumulative_returns, annualized_returns = backtest(df)
        
        print(f"\n回测结果:")
        print(f"累计收益率: {cumulative_returns:.2f}%")
        print(f"年化收益率: {annualized_returns:.2f}%")
        
        print("\n最近交易信号:")
        recent_signals = df[['time', 'close', 'Signal']].tail(5)
        print(recent_signals)
        
        print("\n通过AI进行市场分析...")
        ai_analysis = analyze_market_with_ai(df)
        print(ai_analysis)
        
        # 创建结果目录
        os.makedirs("results", exist_ok=True)
        
        # 保存结果到CSV
        result_file = f"results/{symbol.replace('.', '_')}_analysis.csv"
        df.to_csv(result_file, index=False)
        print(f"分析结果已保存到 {result_file}")
        
    except Exception as e:
        print(f"发生错误: {e}")
        import traceback
        traceback.print_exc()
    finally:
        # QuoteContext不再有close方法
        pass

if __name__ == "__main__":
    import asyncio
    asyncio.run(main()) 