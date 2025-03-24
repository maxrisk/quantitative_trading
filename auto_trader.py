#!/usr/bin/env python
# -*- coding: utf-8 -*-

import os
import sys
import asyncio
import time
import json
import traceback
import pandas as pd
import numpy as np
from datetime import datetime, timedelta
from dotenv import load_dotenv
from longport.openapi import Config, TradeContext, QuoteContext, OrderSide, OrderType, TimeInForceType, Order, OrderStatus
import logging
import decimal
import enum

# 配置日志
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(name)s - %(levelname)s - %(message)s',
    handlers=[
        logging.StreamHandler()
    ]
)
logger = logging.getLogger("AutoTrader")

# 加载环境变量
load_dotenv()

# 长桥API配置
app_key = os.getenv('LONGPORT_APP_KEY')
app_secret = os.getenv('LONGPORT_APP_SECRET')
access_token = os.getenv('LONGPORT_ACCESS_TOKEN')

config = Config(app_key=app_key, app_secret=app_secret, access_token=access_token)

# 交易配置
TRADE_CONFIG = {
    "mode": "paper",  # "paper"(模拟交易) 或 "live"(实盘交易)
    "capital_limit": 100000,  # 交易资金限制
    "max_positions": 5,  # 最大持仓股票数量
    "position_size": 0.2,  # 单个仓位占总资金的比例 (20%)
    "stop_loss": 0.05,  # 止损比例 (5%)
    "take_profit": 0.15,  # 止盈比例 (15%)
    "trading_hours": {
        "HK": {"start": "09:30", "end": "16:00"}  # 香港市场交易时间
    }
}

class AutoTrader:
    """自动化交易类，负责执行交易信号和管理持仓"""
    
    def __init__(self, config):
        self.config = config
        self.trade_config = TRADE_CONFIG
        self.positions = {}  # 当前持仓
        self.orders = {}  # 当前订单
        self.trade_ctx = None
        self.quote_ctx = None
        self.is_trading = False
        
        # 创建结果目录
        os.makedirs("results", exist_ok=True)
    
    async def initialize(self):
        """初始化交易和行情上下文"""
        try:
            logger.info("开始初始化交易和行情上下文...")
            # 确保清除旧的上下文对象，避免状态混淆
            self.trade_ctx = None
            self.quote_ctx = None
            
            # 创建新的上下文对象
            try:
                self.trade_ctx = TradeContext(self.config)
                logger.info("成功创建交易上下文")
            except Exception as e:
                logger.error(f"创建交易上下文失败: {e}")
                logger.error(traceback.format_exc())
                return False
                
            try:
                self.quote_ctx = QuoteContext(self.config)
                logger.info("成功创建行情上下文")
            except Exception as e:
                logger.error(f"创建行情上下文失败: {e}")
                logger.error(traceback.format_exc())
                return False
            
            # 获取账户余额信息作为账户确认
            logger.info("尝试获取账户余额信息...")
            try:
                account_balance_response = self.trade_ctx.account_balance()
                logger.info(f"账户余额响应类型: {type(account_balance_response)}")
                
                # 打印完整响应内容以便调试
                logger.info(f"账户余额响应内容: {account_balance_response}")
                
                if not account_balance_response:
                    logger.error("账户余额响应为空")
                    return False
                
                # 检查response是否有balances属性
                if not hasattr(account_balance_response, 'balances'):
                    logger.info("账户余额响应使用的是列表格式而非balances属性")
                    # 尝试直接使用响应对象
                    if isinstance(account_balance_response, list) and len(account_balance_response) > 0:
                        logger.info(f"账户余额响应是一个列表，长度为 {len(account_balance_response)}")
                        
                        # 获取第一个账户余额信息
                        first_account = account_balance_response[0]
                        logger.info(f"账户币种: {first_account.currency}")
                        logger.info(f"账户总现金: {first_account.total_cash}")
                        
                        # 遍历现金信息
                        if hasattr(first_account, 'cash_infos') and first_account.cash_infos:
                            for cash_info in first_account.cash_infos:
                                logger.info(f"币种 {cash_info.currency}: 可用现金 {cash_info.available_cash}")
                        
                        # 初始化成功
                        return True
                    else:
                        logger.warning("账户余额响应不是列表格式也没有balances属性")
                        return False
                else:
                    balances = account_balance_response.balances
                    if not balances:
                        logger.warning("balances属性为空")
                        return False
                    
                    # 遍历余额信息
                    for balance in balances:
                        logger.info(f"账户币种: {balance.currency}")
                        logger.info(f"账户总现金: {balance.total_cash}")
                    
                    # 初始化成功
                    return True
            except Exception as e:
                logger.error(f"获取账户余额时出错: {e}")
                import traceback
                logger.error(traceback.format_exc())
                return False
            
            # 获取账户资金流水信息
            logger.info("尝试获取账户资金流水...")
            try:
                end_at = datetime.now()
                start_at = end_at - timedelta(days=7)
                cash_flow_response = self.trade_ctx.cash_flow(start_at=start_at, end_at=end_at)
                logger.info(f"资金流水响应类型: {type(cash_flow_response)}")
                
                if cash_flow_response:
                    logger.info(f"账户资金流水获取成功")
                else:
                    logger.warning("资金流水响应为空，但继续执行")
            except Exception as e:
                logger.warning(f"获取资金流水时出错，但继续执行: {e}")
            
            # 获取当前持仓
            try:
                await self.update_positions()
            except Exception as e:
                logger.warning(f"更新持仓信息时出错，但继续执行: {e}")
            
            logger.info("交易和行情上下文初始化成功")
            return True
        except Exception as e:
            logger.error(f"初始化交易上下文失败: {e}")
            import traceback
            logger.error(traceback.format_exc())
            return False
    
    async def update_positions(self):
        """更新当前持仓"""
        try:
            logger.info("更新持仓信息...")
            
            # 检查trade_ctx是否有效
            if self.trade_ctx is None:
                logger.error("交易上下文为空，尝试重新初始化...")
                if not await self.initialize():
                    logger.error("初始化交易上下文失败，无法更新持仓")
                    return
            
            stock_positions_response = self.trade_ctx.stock_positions()
            logger.info(f"持仓响应类型: {type(stock_positions_response)}")
            
            # 打印完整响应内容以便调试
            logger.info(f"持仓响应内容: {stock_positions_response}")
            
            # 如果响应是空的，就清空持仓
            if not stock_positions_response:
                logger.info("没有持仓信息，清空持仓字典")
                self.positions = {}
                return
            
            # 创建一个新的空字典来存储持仓信息，确保完全替换而不是叠加
            positions_dict = {}
            
            # 检查response的类型和结构 - 新版API可能使用不同结构
            if hasattr(stock_positions_response, 'channels'):
                # 新版API结构: 通过channels获取持仓
                channels = stock_positions_response.channels
                if channels:
                    for channel in channels:
                        if hasattr(channel, 'positions') and channel.positions:
                            for position in channel.positions:
                                if hasattr(position, 'symbol'):
                                    symbol = position.symbol
                                    quantity = position.quantity if hasattr(position, 'quantity') else 0
                                    cost_price = float(position.cost_price) if hasattr(position, 'cost_price') and position.cost_price is not None else 0.0
                                    
                                    # 先设置默认值
                                    current_price = cost_price  # 默认使用成本价作为当前价格
                                    market_value = float(float(quantity) * cost_price)
                                    
                                    positions_dict[symbol] = {
                                        "symbol": symbol,
                                        "quantity": quantity,
                                        "cost_price": cost_price,
                                        "current_price": current_price,
                                        "market_value": market_value,
                                        "entry_time": datetime.now().strftime("%Y-%m-%d %H:%M:%S")
                                    }
                                    logger.info(f"更新持仓: {symbol} - 数量: {quantity}, 成本价: {cost_price}")
            elif hasattr(stock_positions_response, 'positions'):
                # 如果有positions属性
                positions = stock_positions_response.positions
                if positions:
                    for position in positions:
                        symbol = position.symbol
                        positions_dict[symbol] = {
                            "symbol": symbol,
                            "quantity": position.quantity,
                            "cost_price": float(position.avg_price) if hasattr(position, 'avg_price') and position.avg_price is not None else 0.0,
                            "current_price": float(position.last_done) if hasattr(position, 'last_done') and position.last_done is not None else 0.0,
                            "market_value": float(position.market_value) if hasattr(position, 'market_value') and position.market_value is not None else 0.0,
                            "entry_time": datetime.now().strftime("%Y-%m-%d %H:%M:%S")
                        }
                        logger.info(f"更新持仓: {symbol} - 数量: {position.quantity}, 成本价: {positions_dict[symbol]['cost_price']}")
            elif isinstance(stock_positions_response, list):
                # 如果是列表，直接迭代
                for position in stock_positions_response:
                    if hasattr(position, 'symbol'):
                        symbol = position.symbol
                        quantity = position.quantity if hasattr(position, 'quantity') else 0
                        cost_price = float(position.avg_price) if hasattr(position, 'avg_price') and position.avg_price is not None else 0.0
                        current_price = float(position.last_done) if hasattr(position, 'last_done') and position.last_done is not None else 0.0
                        market_value = float(position.market_value) if hasattr(position, 'market_value') and position.market_value is not None else 0.0
                        
                        positions_dict[symbol] = {
                            "symbol": symbol,
                            "quantity": quantity,
                            "cost_price": cost_price,
                            "current_price": current_price,
                            "market_value": market_value,
                            "entry_time": datetime.now().strftime("%Y-%m-%d %H:%M:%S")
                        }
                        logger.info(f"更新持仓: {symbol} - 数量: {quantity}, 成本价: {cost_price}")
            
            # 完全替换类的持仓信息，而不是合并或叠加
            self.positions = positions_dict
            logger.info(f"持仓更新完成，共 {len(self.positions)} 个持仓: {list(self.positions.keys())}")
            
            # 手动调用一次保存交易状态，确保持仓信息写入文件
            await self.save_trading_state()
            
        except Exception as e:
            logger.error(f"更新持仓信息失败: {e}")
            import traceback
            logger.error(traceback.format_exc())
    
    async def update_market_data(self):
        """更新市场数据，包括当前持仓的股票价格"""
        try:
            if not self.positions:
                logger.info("当前没有持仓，跳过市场数据更新")
                return True
            
            # 检查quote_ctx是否有效
            if self.quote_ctx is None:
                logger.error("行情上下文为空，尝试重新初始化...")
                if not await self.initialize():
                    logger.error("初始化行情上下文失败，无法更新市场数据")
                    return False
            
            symbols = list(self.positions.keys())
            logger.info(f"获取行情数据: {symbols}")
            
            # 分批处理股票，防止一次请求过多
            batch_size = 10
            updated_count = 0
            
            for i in range(0, len(symbols), batch_size):
                batch_symbols = symbols[i:i+batch_size]
                try:
                    quote_response = self.quote_ctx.quote(batch_symbols)
                    logger.debug(f"行情响应类型: {type(quote_response)}")
                    
                    # 检查响应格式并更新价格
                    if hasattr(quote_response, 'quotes'):
                        quotes = quote_response.quotes
                        for quote in quotes:
                            if hasattr(quote, 'symbol') and quote.symbol in self.positions:
                                if hasattr(quote, 'last_done') and quote.last_done is not None:
                                    try:
                                        # 格式转换
                                        price = float(quote.last_done)
                                        if price > 0:
                                            self.positions[quote.symbol]["current_price"] = price
                                            logger.debug(f"更新价格: {quote.symbol} - {price}")
                                            updated_count += 1
                                        else:
                                            logger.warning(f"{quote.symbol} 的价格为零或负值: {price}")
                                    except (ValueError, TypeError) as e:
                                        logger.error(f"转换 {quote.symbol} 的价格时出错: {e}")
                                else:
                                    logger.warning(f"{quote.symbol} 缺少last_done属性")
                    elif isinstance(quote_response, list):
                        # 如果是列表，直接迭代
                        for quote in quote_response:
                            if hasattr(quote, 'symbol') and quote.symbol in self.positions:
                                if hasattr(quote, 'last_done') and quote.last_done is not None:
                                    try:
                                        # 格式转换
                                        price = float(quote.last_done)
                                        if price > 0:
                                            self.positions[quote.symbol]["current_price"] = price
                                            logger.debug(f"更新价格: {quote.symbol} - {price}")
                                            updated_count += 1
                                        else:
                                            logger.warning(f"{quote.symbol} 的价格为零或负值: {price}")
                                    except (ValueError, TypeError) as e:
                                        logger.error(f"转换 {quote.symbol} 的价格时出错: {e}")
                                else:
                                    logger.warning(f"{quote.symbol} 缺少last_done属性")
                    elif isinstance(quote_response, dict):
                        # 如果是字典，尝试从不同键中提取数据
                        if 'quotes' in quote_response:
                            quotes = quote_response['quotes']
                            for quote in quotes:
                                if 'symbol' in quote and quote['symbol'] in self.positions:
                                    if 'last_done' in quote and quote['last_done'] is not None:
                                        try:
                                            price = float(quote['last_done'])
                                            if price > 0:
                                                self.positions[quote['symbol']]["current_price"] = price
                                                logger.debug(f"更新价格: {quote['symbol']} - {price}")
                                                updated_count += 1
                                            else:
                                                logger.warning(f"{quote['symbol']} 的价格为零或负值: {price}")
                                        except (ValueError, TypeError) as e:
                                            logger.error(f"转换 {quote['symbol']} 的价格时出错: {e}")
                                    else:
                                        logger.warning(f"{quote['symbol']} 缺少last_done键")
                    else:
                        logger.warning(f"未知的行情响应格式: {type(quote_response)}")
                except Exception as e:
                    logger.error(f"获取批次 {batch_symbols} 的行情数据时出错: {e}")
                    import traceback
                    logger.error(traceback.format_exc())
                    continue
            
            if updated_count > 0:
                logger.info(f"成功更新了 {updated_count}/{len(symbols)} 个股票的价格")
                return True
            else:
                logger.warning(f"未能更新任何股票的价格")
                return False
        except Exception as e:
            logger.error(f"更新市场数据失败: {e}")
            import traceback
            logger.error(traceback.format_exc())
            return False
    
    async def check_trading_hours(self, symbol):
        """检查当前是否在交易时间"""
        # 获取股票市场信息
        market = "HK" if symbol.endswith("HK") else "US"
        trading_hours = self.trade_config["trading_hours"].get(market)
        
        if not trading_hours:
            logger.warning(f"未找到市场 {market} 的交易时间配置")
            return False
        
        now = datetime.now()
        now_time = now.strftime("%H:%M")
        
        if trading_hours["start"] <= now_time <= trading_hours["end"]:
            return True
        else:
            logger.info(f"当前不在 {market} 交易时间内: {trading_hours['start']} - {trading_hours['end']}")
            return False
    
    async def process_signals(self, signals_df):
        """处理交易信号并执行交易"""
        if not self.is_trading:
            logger.info("交易未启动，跳过信号处理")
            return
        
        if not signals_df.empty:
            logger.info(f"处理信号DataFrame: {signals_df}")
            for index, row in signals_df.iterrows():
                try:
                    symbol = row["symbol"]
                    signal = row["Trade_Signal"]
                    logger.info(f"处理 {symbol} 的信号: {signal}")
                    price = row["close"]
                    
                    # 检查交易时间
                    if not await self.check_trading_hours(symbol):
                        logger.info(f"{symbol} 当前不在交易时间，跳过")
                        continue
                    
                    if signal == 1:  # 买入信号
                        logger.info(f"发现买入信号，准备买入 {symbol}")
                        await self.place_buy_order(symbol, price)
                    elif signal == -1:  # 卖出信号
                        logger.info(f"发现卖出信号，准备卖出 {symbol}")
                        await self.place_sell_order(symbol, price)
                    else:
                        logger.info(f"信号值为 {signal}，不执行交易操作")
                except KeyError as e:
                    logger.error(f"处理信号时发生KeyError: {e}，可用的列名: {row.index.tolist()}")
                except Exception as e:
                    logger.error(f"处理信号时发生错误: {e}")
                    import traceback
                    logger.error(traceback.format_exc())
    
    async def place_buy_order(self, symbol, price):
        """下买入订单"""
        try:
            # 检查trade_ctx和quote_ctx是否有效
            if self.trade_ctx is None or self.quote_ctx is None:
                logger.error("交易或行情上下文为空，尝试重新初始化...")
                if not await self.initialize():
                    logger.error("初始化交易和行情上下文失败，无法执行买入操作")
                    return False
            
            # 检查是否已持有该股票
            if symbol in self.positions:
                logger.info(f"已持有 {symbol}，跳过买入")
                return False
            
            # 检查持仓数量是否达到上限
            if len(self.positions) >= self.trade_config["max_positions"]:
                logger.info(f"持仓已达到上限 ({self.trade_config['max_positions']}只)，跳过买入")
                return False
                
            # 检查是否已经有未完成的买入订单
            has_pending_order = False
            for order_id, order in self.orders.items():
                if (order["symbol"] == symbol and 
                    order["side"] == "Buy" and 
                    not any(s in str(order["status"]).upper() for s in ["FILLED", "CANCELED", "REJECTED"])):
                    has_pending_order = True
                    logger.info(f"已有未完成的买入订单 {order_id} 状态为 {order['status']}，跳过重复下单")
                    break
                    
            if has_pending_order:
                return False
            
            # 获取账户现金
            logger.info("尝试获取账户现金...")
            try:
                account_balance_response = self.trade_ctx.account_balance()
                logger.info(f"账户余额响应类型: {type(account_balance_response)}")
                
                # 初始化可用现金变量
                available_cash = 0
                
                # 检查response格式并提取可用现金
                if hasattr(account_balance_response, 'balances'):
                    balances = account_balance_response.balances
                    if balances and len(balances) > 0:
                        # 使用total_cash
                        available_cash = balances[0].total_cash * self.trade_config["position_size"]
                        logger.info(f"从balances属性获取可用现金: {available_cash}")
                elif isinstance(account_balance_response, list) and len(account_balance_response) > 0:
                    # 如果是列表，使用第一个账户的total_cash
                    first_account = account_balance_response[0]
                    # 首先尝试使用total_cash - 将Decimal类型转换为float
                    available_cash = float(first_account.total_cash) * self.trade_config["position_size"]
                    logger.info(f"从账户列表获取总现金: {first_account.total_cash}, 可用现金: {available_cash}")
                    
                    # 如果需要特定币种的现金，可以从cash_infos中找出
                    if hasattr(first_account, 'cash_infos') and first_account.cash_infos:
                        # 找到与symbol匹配的币种
                        currency = "HKD" if symbol.endswith("HK") else "USD"
                        for cash_info in first_account.cash_infos:
                            if cash_info.currency == currency:
                                available_cash = float(cash_info.available_cash) * self.trade_config["position_size"]
                                logger.info(f"从cash_infos中找到{currency}币种的可用现金: {available_cash}")
                                break
                    
                    logger.info(f"最终确定的可用现金: {available_cash}")
                else:
                    logger.warning("无法从账户余额响应中获取可用现金")
                    return False
                
                if available_cash <= 0:
                    logger.info(f"可用现金不足: {available_cash}")
                    return False
            except Exception as e:
                logger.error(f"获取账户余额失败: {e}")
                import traceback
                logger.error(traceback.format_exc())
                return False
            
            # 获取股票的交易单位（手数）
            lot_size = 100  # 默认手数，港股一般是100股/手
            try:
                # 尝试获取股票的静态信息来确定实际手数
                static_info = self.quote_ctx.static_info([symbol])
                logger.info(f"获取股票静态信息: {static_info}")
                
                if hasattr(static_info, 'infos') and static_info.infos:
                    for info in static_info.infos:
                        if hasattr(info, 'lot_size') and info.lot_size > 0:
                            lot_size = info.lot_size
                            logger.info(f"获取到 {symbol} 的手数: {lot_size}")
                            break
                elif isinstance(static_info, list) and len(static_info) > 0:
                    for info in static_info:
                        if hasattr(info, 'lot_size') and info.lot_size > 0:
                            lot_size = info.lot_size
                            logger.info(f"获取到 {symbol} 的手数: {lot_size}")
                            break
            except Exception as e:
                logger.warning(f"获取股票手数失败，使用默认值 {lot_size}: {e}")
            
            # 计算可买入数量（确保是手数的整数倍）
            quantity = int(available_cash / price / lot_size) * lot_size
            
            if quantity <= 0:
                logger.info(f"资金不足或计算可买入数量为零，无法买入 {symbol}")
                return False
            
            logger.info(f"计算得出可买入数量: {quantity} 股（{quantity/lot_size}手，每手{lot_size}股）")
            
            # 下买入订单
            if self.trade_config["mode"] == "paper":
                # 修改为实际调用接口下单
                logger.info(f"[模拟] 买入 {symbol}: {quantity} 股，价格: {price}")
                
                try:
                    # 调用接口实际下单
                    submit_order_result = self.trade_ctx.submit_order(
                        symbol=symbol,
                        order_type=OrderType.LO,  # 限价单
                        side=OrderSide.Buy,
                        submitted_price=price,
                        submitted_quantity=quantity,
                        time_in_force=TimeInForceType.Day
                    )
                    
                    order_id = submit_order_result.order_id
                    logger.info(f"[模拟] 买入订单已提交: {order_id}")
                    
                    # 记录订单
                    self.orders[order_id] = {
                        "symbol": symbol,
                        "side": "Buy",
                        "quantity": quantity,
                        "price": float(price),
                        "status": "submitted",  # 使用字符串表示订单状态
                        "time": datetime.now().strftime("%Y-%m-%d %H:%M:%S")
                    }
                    
                    # 主动更新一次持仓信息
                    await self.update_positions()
                    
                    return True
                except Exception as e:
                    logger.error(f"[模拟] 买入订单API调用失败: {e}")
                    return False
            else:
                # 实盘交易模式
                logger.info(f"下单买入 {symbol}: {quantity} 股，价格: {price}")
                
                try:
                    submit_order_result = self.trade_ctx.submit_order(
                        symbol=symbol,
                        order_type=OrderType.LO,  # 限价单
                        side=OrderSide.Buy,
                        submitted_price=price,
                        submitted_quantity=quantity,
                        time_in_force=TimeInForceType.Day
                    )
                    
                    order_id = submit_order_result.order_id
                    logger.info(f"买入订单已提交: {order_id}")
                    
                    # 记录订单
                    self.orders[order_id] = {
                        "symbol": symbol,
                        "side": "Buy",
                        "quantity": quantity,
                        "price": float(price),
                        "status": "submitted",  # 使用字符串表示订单状态
                        "time": datetime.now().strftime("%Y-%m-%d %H:%M:%S")
                    }
                    
                    return True
                except Exception as e:
                    logger.error(f"买入订单提交失败: {e}")
                    return False
        except Exception as e:
            logger.error(f"下买入订单失败: {e}")
            return False
    
    async def place_sell_order(self, symbol, price):
        """下卖出订单"""
        try:
            # 检查trade_ctx是否有效
            if self.trade_ctx is None:
                logger.error("交易上下文为空，尝试重新初始化...")
                if not await self.initialize():
                    logger.error("初始化交易上下文失败，无法执行卖出操作")
                    return False
            
            # 检查是否持有该股票
            if symbol not in self.positions:
                logger.info(f"未持有 {symbol}，跳过卖出")
                return False
                
            # 检查是否已经有未完成的卖出订单
            has_pending_order = False
            for order_id, order in self.orders.items():
                if (order["symbol"] == symbol and 
                    order["side"] == "Sell" and 
                    not any(s in str(order["status"]).upper() for s in ["FILLED", "CANCELED", "REJECTED"])):
                    has_pending_order = True
                    logger.info(f"已有未完成的卖出订单 {order_id} 状态为 {order['status']}，跳过重复下单")
                    break
                    
            if has_pending_order:
                return False
            
            # 获取持仓数量
            quantity = self.positions[symbol]["quantity"]
            
            # 如果是Decimal类型，转换为float
            if isinstance(quantity, decimal.Decimal):
                quantity = float(quantity)
            elif isinstance(quantity, str):
                quantity = float(quantity)
            
            # 下卖出订单
            if self.trade_config["mode"] == "paper":
                # 修改为实际调用接口下单
                logger.info(f"[模拟] 卖出 {symbol}: {quantity} 股，价格: {price}")
                
                try:
                    # 调用接口实际下单
                    submit_order_result = self.trade_ctx.submit_order(
                        symbol=symbol,
                        order_type=OrderType.LO,  # 限价单
                        side=OrderSide.Sell,
                        submitted_price=price,
                        submitted_quantity=quantity,
                        time_in_force=TimeInForceType.Day
                    )
                    
                    order_id = submit_order_result.order_id
                    logger.info(f"[模拟] 卖出订单已提交: {order_id}")
                    
                    # 记录订单
                    self.orders[order_id] = {
                        "symbol": symbol,
                        "side": "Sell",
                        "quantity": quantity,
                        "price": float(price),
                        "status": "submitted",  # 使用字符串表示订单状态
                        "time": datetime.now().strftime("%Y-%m-%d %H:%M:%S")
                    }
                    
                    # 计算收益
                    cost_price = float(self.positions[symbol]["cost_price"])
                    price = float(price)
                    profit = (price - cost_price) * quantity
                    profit_pct = (price / cost_price - 1) * 100
                    
                    logger.info(f"[模拟] 卖出 {symbol} 盈亏: {profit:.2f} ({profit_pct:.2f}%)")
                    
                    # 返回成功
                    return True
                except Exception as e:
                    logger.error(f"[模拟] 卖出订单API调用失败: {e}")
                    # 接口调用失败时，仍然移除模拟持仓
                    
                    # 计算收益
                    cost_price = float(self.positions[symbol]["cost_price"])
                    price = float(price)
                    profit = (price - cost_price) * quantity
                    profit_pct = (price / cost_price - 1) * 100
                    
                    logger.info(f"[模拟] 卖出 {symbol} 盈亏: {profit:.2f} ({profit_pct:.2f}%)")
                    
                    # 移除持仓
                    del self.positions[symbol]
                    
                    # 返回失败
                    return False
            else:
                # 实盘交易模式
                logger.info(f"下单卖出 {symbol}: {quantity} 股，价格: {price}")
                
                try:
                    submit_order_result = self.trade_ctx.submit_order(
                        symbol=symbol,
                        order_type=OrderType.LO,  # 限价单
                        side=OrderSide.Sell,
                        submitted_price=price,
                        submitted_quantity=quantity,
                        time_in_force=TimeInForceType.Day
                    )
                    
                    order_id = submit_order_result.order_id
                    logger.info(f"卖出订单已提交: {order_id}")
                    
                    # 记录订单
                    self.orders[order_id] = {
                        "symbol": symbol,
                        "side": "Sell",
                        "quantity": quantity,
                        "price": float(price),
                        "status": "submitted",  # 使用字符串表示订单状态
                        "time": datetime.now().strftime("%Y-%m-%d %H:%M:%S")
                    }
                    
                    return True
                except Exception as e:
                    logger.error(f"卖出订单提交失败: {e}")
                    return False
        except Exception as e:
            logger.error(f"下卖出订单失败: {e}")
            return False
    
    async def update_orders(self):
        """更新订单状态"""
        try:
            # 检查trade_ctx是否有效
            if self.trade_ctx is None:
                logger.error("交易上下文为空，尝试重新初始化...")
                if not await self.initialize():
                    logger.error("初始化交易上下文失败，无法更新订单状态")
                    return False
            
            # 获取当前时间，用于计算订单提交时长
            current_time = datetime.now()
            
            # 将get_orders改为today_orders或history_orders方法
            try:
                # 尝试使用today_orders方法获取当天订单
                orders_response = self.trade_ctx.today_orders()
                logger.debug(f"获取今日订单状态响应: {type(orders_response)}")
            except AttributeError:
                # 如果today_orders不存在，尝试使用可能的其他方法
                try:
                    # 尝试使用无参数的orders方法
                    orders_response = self.trade_ctx.orders()
                    logger.debug(f"获取订单状态响应(orders方法): {type(orders_response)}")
                except AttributeError:
                    # 最后尝试使用history_orders方法
                    end_at = datetime.now()
                    start_at = end_at - timedelta(days=7)  # 获取过去7天的订单
                    orders_response = self.trade_ctx.history_orders(start_at=start_at, end_at=end_at)
                    logger.debug(f"获取历史订单状态响应: {type(orders_response)}")
            
            # 处理不同格式的响应数据
            orders_data = None
            if hasattr(orders_response, 'orders'):
                orders_data = orders_response.orders
            elif isinstance(orders_response, list):
                orders_data = orders_response
            elif isinstance(orders_response, dict) and 'orders' in orders_response:
                orders_data = orders_response['orders']
            else:
                logger.warning(f"未知的订单响应格式: {type(orders_response)}")
                return False
                
            # 更新订单信息
            updated_count = 0
            completed_orders = []  # 记录已完成的订单
            
            for order in orders_data:
                order_id = order.order_id if hasattr(order, 'order_id') else order.get('order_id')
                if not order_id:
                    continue
                    
                status = order.status if hasattr(order, 'status') else order.get('status')
                # 确保订单状态是字符串
                if hasattr(status, "__class__") and status.__class__.__name__ == "OrderStatus":
                    status = str(status)
                elif isinstance(status, (enum.Enum, OrderStatus)):
                    status = str(status)
                    
                if order_id in self.orders:
                    old_status = self.orders[order_id]["status"]
                    self.orders[order_id]["status"] = status
                    updated_count += 1
                    
                    # 记录状态变化
                    if old_status != status:
                        logger.info(f"订单 {order_id} 状态从 {old_status} 变更为 {status}")
                    
                    # 处理已完成的订单
                    if (isinstance(status, str) and any(s in status.upper() for s in ["FILLED", "CANCELED", "REJECTED"])) or \
                       (hasattr(status, "value") and any(s in str(status.value).upper() for s in ["FILLED", "PARTIALLY_FILLED", "CANCELED", "REJECTED"])):
                        
                        # 如果订单已成交，更新持仓信息
                        if any(s in str(status).upper() for s in ["FILLED", "PARTIALLY_FILLED"]):
                            logger.info(f"订单 {order_id} 已成交或部分成交: {status}")
                            await self.update_positions()
                            
                        # 将已完成的订单添加到列表中，稍后处理    
                        completed_orders.append(order_id)
                        
            # 检查长时间未成交的订单，可能需要取消重新下单
            orders_to_cancel = []
            for order_id, order in self.orders.items():
                # 跳过已完成的订单
                if any(s in str(order["status"]).upper() for s in ["FILLED", "CANCELED", "REJECTED"]):
                    continue
                    
                # 订单时间超过24小时，可以考虑取消
                try:
                    order_time = datetime.strptime(order["time"], "%Y-%m-%d %H:%M:%S")
                    time_diff = (current_time - order_time).total_seconds() / 3600  # 小时
                    
                    if time_diff > 24:
                        logger.warning(f"订单 {order_id} 已提交超过24小时仍未成交，考虑取消重新下单")
                        orders_to_cancel.append(order_id)
                except (ValueError, KeyError):
                    # 订单时间格式不正确，跳过
                    continue
                    
            # 取消长时间未成交的订单
            for order_id in orders_to_cancel:
                try:
                    symbol = self.orders[order_id]["symbol"]
                    logger.info(f"尝试取消长时间未成交的订单: {order_id}, 股票: {symbol}")
                    
                    # 调用API取消订单
                    cancel_result = self.trade_ctx.cancel_order(order_id=order_id)
                    if cancel_result:
                        logger.info(f"成功取消订单 {order_id}")
                        self.orders[order_id]["status"] = "CANCELED"
                    else:
                        logger.warning(f"取消订单 {order_id} 失败")
                except Exception as e:
                    logger.error(f"取消订单 {order_id} 时出错: {e}")
                    
            # 处理已完成的订单
            for order_id in completed_orders:
                # 可以考虑删除已完成的订单，或者移动到历史订单中
                # 这里先添加一个标记，表示已处理
                if order_id in self.orders:
                    self.orders[order_id]["processed"] = True
                    
            logger.info(f"更新了 {updated_count} 个订单状态，完成处理 {len(completed_orders)} 个订单")
            
            # 保存更新后的交易状态
            await self.save_trading_state()
            
            return True
        except Exception as e:
            logger.error(f"更新订单状态失败: {e}")
            logger.error(f"订单更新失败，将在下一次周期重试")
            return False
    
    async def check_stop_loss_take_profit(self):
        """检查止损止盈条件"""
        try:
            if not self.positions or len(self.positions) == 0:
                logger.info("没有持仓，跳过止损止盈检查")
                return
                
            # 检查trade_ctx和quote_ctx是否有效
            if self.trade_ctx is None or self.quote_ctx is None:
                logger.error("交易或行情上下文为空，尝试重新初始化...")
                if not await self.initialize():
                    logger.error("初始化交易和行情上下文失败，无法执行止损止盈检查")
                    return
            
            logger.info(f"开始检查 {len(self.positions)} 个持仓的止损止盈")
            
            # 获取所有持仓的股票代码，用于更新行情数据
            symbols = list(self.positions.keys())
            logger.info(f"获取行情数据: {symbols}")
            
            # 更新行情数据
            update_success = await self.update_market_data()
            if not update_success:
                logger.warning("获取行情数据失败，无法执行止损止盈检查")
                return
            
            logger.info(f"成功更新了 {len(symbols)}/{len(symbols)} 个股票的价格")
            
            # 止损止盈阈值
            stop_loss_threshold = -0.05  # 亏损5%触发止损
            take_profit_threshold = 0.10  # 盈利10%触发止盈
            
            triggered_operations = []  # 记录触发的操作
            
            # 检查每个持仓
            for symbol, position in self.positions.items():
                try:
                    # 如果没有拿到当前价格，跳过
                    if "current_price" not in position or position["current_price"] is None:
                        logger.warning(f"股票 {symbol} 缺少当前价格信息，跳过止损止盈检查")
                        continue
                    
                    cost_price = position.get("cost_price", 0)
                    current_price = position.get("current_price", 0)
                    quantity = position.get("quantity", 0)
                    
                    # 确保价格和数量为数值类型
                    try:
                        # 如果是Decimal类型，转换为float
                        if isinstance(cost_price, decimal.Decimal):
                            cost_price = float(cost_price)
                        elif isinstance(cost_price, str):
                            cost_price = float(cost_price)
                            
                        if isinstance(current_price, decimal.Decimal):
                            current_price = float(current_price)
                        elif isinstance(current_price, str):
                            current_price = float(current_price)
                            
                        if isinstance(quantity, decimal.Decimal):
                            quantity = float(quantity)
                        elif isinstance(quantity, str):
                            quantity = float(quantity)
                    except (ValueError, TypeError) as e:
                        logger.warning(f"转换 {symbol} 的价格或数量时出错: {e}")
                        continue
                    
                    # 检查成本价是否有效
                    if cost_price <= 0:
                        logger.warning(f"股票 {symbol} 的成本价无效: {cost_price}")
                        continue
                    
                    # 计算盈亏
                    profit_pct = (current_price - cost_price) / cost_price
                    profit_amount = (current_price - cost_price) * quantity
                    
                    # 记录当前盈亏情况
                    logger.info(f"股票 {symbol}: 成本价 {cost_price:.2f}, 现价 {current_price:.2f}, 数量 {quantity}, 盈亏 {profit_pct:.2%} (¥{profit_amount:.2f})")
                    
                    # 检查止损条件
                    if profit_pct <= stop_loss_threshold:
                        logger.warning(f"股票 {symbol} 触发止损条件! 盈亏 {profit_pct:.2%} (¥{profit_amount:.2f})")
                        
                        # 执行卖出操作
                        sell_result = await self.place_sell_order(symbol, current_price)
                        if sell_result:
                            triggered_operations.append(f"止损卖出 {symbol}, 价格 {current_price:.2f}")
                            logger.info(f"已执行止损卖出 {symbol}")
                        else:
                            logger.error(f"止损卖出 {symbol} 失败")
                    
                    # 检查止盈条件
                    if profit_pct >= take_profit_threshold:
                        logger.warning(f"股票 {symbol} 触发止盈条件! 盈亏 {profit_pct:.2%} (¥{profit_amount:.2f})")
                        
                        # 执行卖出操作
                        sell_result = await self.place_sell_order(symbol, current_price)
                        if sell_result:
                            triggered_operations.append(f"止盈卖出 {symbol}, 价格 {current_price:.2f}")
                            logger.info(f"已执行止盈卖出 {symbol}")
                        else:
                            logger.error(f"止盈卖出 {symbol} 失败")
                except Exception as e:
                    logger.error(f"检查 {symbol} 的止损止盈时出错: {e}")
                    logger.error(f"Traceback (most recent call last):\n{traceback.format_exc()}")
            
            # 汇总止损止盈结果
            if triggered_operations:
                logger.info(f"本次检查触发了以下操作: {', '.join(triggered_operations)}")
            else:
                logger.info("没有触发任何止损止盈操作")
            
            return len(triggered_operations) > 0
        except Exception as e:
            logger.error(f"检查止损止盈失败: {e}")
            logger.error(f"Traceback (most recent call last):\n{traceback.format_exc()}")
            return False
    
    async def start_trading(self):
        """初始化并启动交易系统"""
        logger.info(f"启动交易系统 (模式: {self.trade_config['mode']})")
        
        # 初始化交易上下文
        if not await self.initialize():
            logger.error("初始化失败，无法启动交易")
            return False
            
        # 强制更新最新持仓状态
        await self.update_positions()
        
        # 更新订单状态
        await self.update_orders()
        
        # 设置交易标志为开启
        self.is_trading = True
        
        # 保存交易状态
        await self.save_trading_state()
        
        logger.info("交易系统已启动，准备接收交易信号")
        return True
    
    async def stop_trading(self):
        """停止自动交易"""
        logger.info("停止自动交易")
        self.is_trading = False
        
        # 关闭连接
        # 新版API不再需要关闭连接
        pass
    
    async def save_trading_state(self):
        """保存交易状态"""
        # 定义Decimal编码器
        class DecimalEncoder(json.JSONEncoder):
            def default(self, o):
                if isinstance(o, decimal.Decimal):
                    return float(o)
                # 处理OrderStatus枚举
                elif hasattr(o, "__class__") and o.__class__.__name__ == "OrderStatus":
                    return str(o)
                # 处理其他枚举类型
                elif isinstance(o, (enum.Enum, OrderStatus)):
                    return str(o)
                return super().default(o)
                
        state = {
            "positions": self.positions,
            "orders": self.orders,
            "is_trading": self.is_trading,
            "timestamp": datetime.now().strftime("%Y-%m-%d %H:%M:%S")
        }
        
        with open("results/trading_state.json", "w") as f:
            json.dump(state, f, indent=4, cls=DecimalEncoder)
        
        logger.info("交易状态已保存")
    
    async def load_trading_state(self):
        """加载交易状态"""
        state_file = "results/trading_state.json"
        if os.path.exists(state_file):
            try:
                with open(state_file, "r") as f:
                    state = json.load(f)
                
                # 临时保存订单信息和交易状态
                self.orders = state.get("orders", {})
                self.is_trading = state.get("is_trading", False)
                
                # 不加载持仓信息，而是通过API获取最新持仓
                logger.info(f"已加载交易状态 (订单: {len(self.orders)} 个)")
                
                # 立即通过API更新持仓
                await self.update_positions()
                
                return True
            except Exception as e:
                logger.error(f"加载交易状态失败: {e}")
                return False
        else:
            logger.info("未找到保存的交易状态")
            return False
    
    def get_trading_summary(self):
        """获取交易摘要"""
        summary = {
            "positions": len(self.positions),
            "position_details": self.positions,
            "orders": len(self.orders),
            "is_trading": self.is_trading,
            "mode": self.trade_config["mode"]
        }
        
        return summary

async def generate_signals_from_data(data_file):
    """从数据文件中生成交易信号"""
    try:
        if not os.path.exists(data_file):
            logger.error(f"数据文件不存在: {data_file}")
            return pd.DataFrame()
            
        if os.path.getsize(data_file) == 0:
            logger.error(f"数据文件为空: {data_file}")
            return pd.DataFrame()
            
        logger.info(f"从文件 {data_file} 读取交易信号")
        
        # 提取股票代码
        symbol = os.path.basename(data_file).replace("_analysis.csv", "").replace("_", ".")
        
        # 读取CSV文件
        try:
            df = pd.read_csv(data_file)
        except pd.errors.EmptyDataError:
            logger.error(f"数据文件 {data_file} 为空或格式错误")
            return pd.DataFrame()
        except pd.errors.ParserError as e:
            logger.error(f"解析数据文件 {data_file} 时出错: {e}")
            return pd.DataFrame()
            
        # 检查文件中是否有必要的列
        required_columns = ['time', 'close']
        missing_columns = [col for col in required_columns if col not in df.columns]
        if missing_columns:
            logger.error(f"数据文件 {data_file} 缺少必要的列: {missing_columns}")
            return pd.DataFrame()
            
        # 检查文件中是否有Signal或Trade_Signal列
        signal_column = None
        if 'Trade_Signal' in df.columns:
            signal_column = 'Trade_Signal'
        elif 'Signal' in df.columns:
            signal_column = 'Signal'
        else:
            logger.error(f"数据文件 {data_file} 中没有找到交易信号列 (Signal 或 Trade_Signal)")
            return pd.DataFrame()
            
        # 检查数据是否为空
        if df.empty:
            logger.warning(f"数据文件 {data_file} 中没有数据")
            return pd.DataFrame()
            
        # 提取最新的交易信号
        try:
            latest_signals = df.tail(1)[['time', 'close', signal_column]].copy()
            
            # 检查信号值是否有效
            if latest_signals[signal_column].isnull().any():
                logger.warning(f"数据文件 {data_file} 的最新信号包含空值")
                # 将空值替换为0（无信号）
                latest_signals[signal_column] = latest_signals[signal_column].fillna(0)
            
            # 为了统一处理，将列名重命名为Trade_Signal
            if signal_column != 'Trade_Signal':
                latest_signals = latest_signals.rename(columns={signal_column: 'Trade_Signal'})
            
            # 确保close列是数值类型
            latest_signals['close'] = pd.to_numeric(latest_signals['close'], errors='coerce')
            if latest_signals['close'].isnull().any():
                logger.warning(f"数据文件 {data_file} 的最新收盘价包含无效值")
                return pd.DataFrame()
            
            # 确保Trade_Signal列是整数类型
            latest_signals['Trade_Signal'] = latest_signals['Trade_Signal'].astype(int)
            
            # 添加股票代码
            latest_signals["symbol"] = symbol
            
            # 记录信号信息
            signal_value = latest_signals.iloc[0]['Trade_Signal']
            signal_type = "买入" if signal_value == 1 else "卖出" if signal_value == -1 else "无操作"
            logger.info(f"股票 {symbol} 的最新信号为 {signal_value} ({signal_type}), 收盘价: {latest_signals.iloc[0]['close']}")
            
            return latest_signals
        except Exception as e:
            logger.error(f"提取 {data_file} 的最新信号时出错: {e}")
            import traceback
            logger.error(traceback.format_exc())
            return pd.DataFrame()
    except Exception as e:
        logger.error(f"从数据文件生成信号时发生错误: {e}")
        import traceback
        logger.error(traceback.format_exc())
        return pd.DataFrame()

async def auto_trading_loop(target_symbol=None, check_interval=10):
    """自动交易主循环
    
    Args:
        target_symbol: 如果指定，则只交易该股票，否则交易所有分析过的股票
        check_interval: 每次交易循环的间隔时间(秒)
    """
    logger.info("初始化自动交易系统...")
    trader = AutoTrader(config)
    
    # 重试计数器和最大重试次数
    retry_count = 0
    max_retries = 3
    retry_interval = 5  # 初始重试间隔(秒)
    
    # 加载之前的交易状态
    try:
        state_loaded = await trader.load_trading_state()
        if not state_loaded:
            logger.warning("无法加载之前的交易状态，将使用空白状态初始化")
    except Exception as e:
        logger.error(f"加载交易状态时出错: {e}")
        import traceback
        logger.error(traceback.format_exc())
    
    # 启动交易系统
    logger.info("启动交易系统...")
    while retry_count < max_retries:
        try:
            if await trader.start_trading():
                logger.info("交易系统启动成功")
                retry_count = 0  # 重置重试计数器
                break
            else:
                retry_count += 1
                logger.error(f"交易系统启动失败 (尝试 {retry_count}/{max_retries})")
                if retry_count < max_retries:
                    wait_time = retry_interval * retry_count  # 指数退避
                    logger.info(f"将在 {wait_time} 秒后重试...")
                    await asyncio.sleep(wait_time)
                else:
                    logger.error("已达到最大重试次数，退出自动交易")
                    return
        except Exception as e:
            retry_count += 1
            logger.error(f"启动交易系统时出错: {e}")
            import traceback
            logger.error(traceback.format_exc())
            if retry_count < max_retries:
                wait_time = retry_interval * retry_count
                logger.info(f"将在 {wait_time} 秒后重试...")
                await asyncio.sleep(wait_time)
            else:
                logger.error("已达到最大重试次数，退出自动交易")
                return
    
    # 创建健康检查计数器
    health_check_counter = 0
    health_check_interval = 6  # 每6个循环执行一次健康检查
    
    try:
        # 运行交易循环
        logger.info("开始交易循环...")
        cycle_count = 0
        
        while trader.is_trading:
            cycle_start_time = time.time()
            cycle_count += 1
            logger.info(f"===== 交易循环 #{cycle_count} 开始 =====")
            
            try:
                # 检查是否需要进行健康检查
                health_check_counter += 1
                if health_check_counter >= health_check_interval:
                    health_check_counter = 0
                    logger.info("执行交易系统健康检查...")
                    
                    # 确保交易上下文仍然有效
                    try:
                        # 简单的健康检查 - 尝试获取账户余额
                        if trader.trade_ctx:
                            account_balance = trader.trade_ctx.account_balance()
                            if account_balance:
                                logger.info("交易上下文健康检查成功")
                            else:
                                logger.warning("交易上下文健康检查返回空响应，尝试重新初始化")
                                if not await trader.initialize():
                                    logger.error("重新初始化交易上下文失败")
                        else:
                            logger.warning("交易上下文不存在，尝试重新初始化")
                            if not await trader.initialize():
                                logger.error("重新初始化交易上下文失败")
                    except Exception as e:
                        logger.error(f"健康检查时出错: {e}")
                        import traceback
                        logger.error(traceback.format_exc())
                        if not await trader.initialize():
                            logger.error("重新初始化交易上下文失败")
                
                # 收集交易信号
                all_signals = pd.DataFrame()
                
                if target_symbol:
                    # 只处理指定股票
                    file_path = os.path.join("results", f"{target_symbol.replace('.', '_')}_analysis.csv")
                    if os.path.exists(file_path):
                        try:
                            signals = await generate_signals_from_data(file_path)
                            all_signals = pd.concat([all_signals, signals])
                            logger.info(f"专注于交易 {target_symbol}")
                        except Exception as e:
                            logger.error(f"处理 {target_symbol} 信号时出错: {e}")
                    else:
                        logger.warning(f"未找到 {target_symbol} 的分析文件：{file_path}")
                else:
                    # 扫描结果目录中的所有分析文件
                    try:
                        result_files = [f for f in os.listdir("results") if f.endswith("_analysis.csv")]
                        logger.info(f"发现 {len(result_files)} 个分析文件")
                        
                        # 从每个文件中提取最新信号
                        for file in result_files:
                            try:
                                file_path = os.path.join("results", file)
                                signals = await generate_signals_from_data(file_path)
                                all_signals = pd.concat([all_signals, signals])
                            except Exception as e:
                                symbol = file.replace("_analysis.csv", "").replace("_", ".")
                                logger.error(f"处理 {symbol} 的信号时出错: {e}")
                                continue
                    except Exception as e:
                        logger.error(f"扫描分析文件时出错: {e}")
                
                # 处理交易信号
                if not all_signals.empty:
                    logger.info(f"发现 {len(all_signals)} 个交易信号")
                    logger.info("开始处理交易信号...")
                    try:
                        await trader.process_signals(all_signals)
                        logger.info("交易信号处理完成")
                    except Exception as e:
                        logger.error(f"处理交易信号时出错: {e}")
                        import traceback
                        logger.error(traceback.format_exc())
                else:
                    logger.info("没有发现新的交易信号")
                
                # 更新订单状态
                logger.info("开始更新订单状态...")
                try:
                    await trader.update_orders()
                    logger.info("订单状态更新完成")
                except Exception as e:
                    logger.error(f"更新订单状态时出错: {e}")
                    import traceback
                    logger.error(traceback.format_exc())
                
                # 检查止损止盈
                logger.info("开始检查止损止盈...")
                try:
                    await trader.check_stop_loss_take_profit()
                    logger.info("止损止盈检查完成")
                except Exception as e:
                    logger.error(f"检查止损止盈时出错: {e}")
                    import traceback
                    logger.error(traceback.format_exc())
                
                # 保存交易状态
                logger.info("开始保存交易状态...")
                try:
                    await trader.save_trading_state()
                    logger.info("交易状态保存完成")
                except Exception as e:
                    logger.error(f"保存交易状态时出错: {e}")
                    import traceback
                    logger.error(traceback.format_exc())
                
                # 计算循环用时并等待到下一次检查
                cycle_end_time = time.time()
                cycle_duration = cycle_end_time - cycle_start_time
                logger.info(f"交易循环耗时: {cycle_duration:.2f} 秒")
                
                # 计算实际需要等待的时间
                wait_time = max(0.1, check_interval - cycle_duration)
                logger.info(f"等待 {wait_time:.2f} 秒后开始下一次检查...")
                await asyncio.sleep(wait_time)
                
            except asyncio.CancelledError:
                logger.info("交易循环被取消")
                break
            except Exception as e:
                logger.error(f"交易循环中发生错误: {e}")
                import traceback
                logger.error(traceback.format_exc())
                # 遇到错误时等待一段时间后继续
                logger.info(f"将在 {retry_interval} 秒后继续...")
                await asyncio.sleep(retry_interval)
            
            logger.info(f"===== 交易循环 #{cycle_count} 结束 =====")
            
    except KeyboardInterrupt:
        logger.info("接收到终止信号，停止交易")
    except Exception as e:
        logger.error(f"交易循环发生意外错误: {e}")
        import traceback
        traceback.print_exc()
    finally:
        # 停止交易并保存状态
        logger.info("清理交易资源...")
        try:
            await trader.stop_trading()
            logger.info("交易系统已停止")
        except Exception as e:
            logger.error(f"停止交易时出错: {e}")
        
        try:
            await trader.save_trading_state()
            logger.info("最终交易状态已保存")
        except Exception as e:
            logger.error(f"保存最终交易状态时出错: {e}")
        
        logger.info("自动交易已完全退出")

async def main():
    """主函数"""
    try:
        # 创建结果目录
        os.makedirs("results", exist_ok=True)
        
        # 运行自动交易循环
        await auto_trading_loop()
    except Exception as e:
        logger.error(f"运行自动交易时发生错误: {e}")

if __name__ == "__main__":
    try:
        asyncio.run(main())
    except KeyboardInterrupt:
        print("程序已被用户终止")
