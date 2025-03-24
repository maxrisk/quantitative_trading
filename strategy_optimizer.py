#!/usr/bin/env python
# -*- coding: utf-8 -*-

import os
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import json
from itertools import product
from datetime import datetime
from dotenv import load_dotenv
from tqdm import tqdm
import time
import multiprocessing as mp  # 添加多进程支持
from functools import partial  # 用于函数参数部分应用

# 加载主模块的函数
from multi_stock_analysis import calculate_indicators, generate_signals, backtest

# 加载环境变量
load_dotenv()

# 创建结果目录
os.makedirs("results/optimizations", exist_ok=True)
os.makedirs("plots/optimizations", exist_ok=True)

# 策略参数范围
PARAM_RANGES = {
    "ma_short": [5, 10],
    "ma_long": [20, 30],
    "rsi_period": [10, 14],
    "rsi_oversold": [25, 30],
    "rsi_overbought": [70, 75]
}

def _evaluate_params(params, df, metric="sharpe_ratio"):
    """评估单个参数组合的性能，用于并行处理"""
    ma_short, ma_long, rsi_period, rsi_oversold, rsi_overbought = params
    
    # 自定义指标计算函数
    def custom_calculate_indicators(df):
        # 计算移动平均线
        df['MA5'] = df['close'].rolling(window=ma_short).mean()
        df['MA10'] = df['close'].rolling(window=ma_long).mean()
        
        # 计算RSI
        delta = df['close'].diff()
        gain = delta.where(delta > 0, 0)
        loss = -delta.where(delta < 0, 0)
        avg_gain = gain.rolling(window=rsi_period).mean()
        avg_loss = loss.rolling(window=rsi_period).mean()
        rs = avg_gain / avg_loss
        df['RSI'] = 100 - (100 / (1 + rs))
        
        # 计算MACD
        ema12 = df['close'].ewm(span=12, adjust=False).mean()
        ema26 = df['close'].ewm(span=26, adjust=False).mean()
        df['MACD'] = ema12 - ema26
        df['Signal_Line'] = df['MACD'].ewm(span=9, adjust=False).mean()
        
        return df
    
    # 自定义信号生成函数
    def custom_generate_signals(df):
        df['Signal'] = 0
        
        # 使用移动平均线交叉
        df.loc[(df['MA5'] > df['MA10']) & (df['MA5'].shift(1) <= df['MA10'].shift(1)), 'Signal'] = 1  # 金叉买入
        df.loc[(df['MA5'] < df['MA10']) & (df['MA5'].shift(1) >= df['MA10'].shift(1)), 'Signal'] = -1  # 死叉卖出
        
        # 使用RSI超买超卖
        df.loc[(df['RSI'] < rsi_oversold), 'Signal'] = 1  # RSI超卖买入
        df.loc[(df['RSI'] > rsi_overbought), 'Signal'] = -1  # RSI超买卖出
        
        # 使用MACD交叉
        df.loc[(df['MACD'] > df['Signal_Line']) & (df['MACD'].shift(1) <= df['Signal_Line'].shift(1)), 'Signal'] = 1  # MACD金叉买入
        df.loc[(df['MACD'] < df['Signal_Line']) & (df['MACD'].shift(1) >= df['Signal_Line'].shift(1)), 'Signal'] = -1  # MACD死叉卖出
        
        return df
    
    try:
        # 应用指标和信号
        df_copy = df.copy()
        df_copy = custom_calculate_indicators(df_copy)
        df_copy = custom_generate_signals(df_copy)
        
        # 进行回测
        df_copy, metrics = backtest(df_copy)
        
        # 计算夏普比率（简化版，假设无风险收益率为0）
        if len(df_copy) > 1:
            sharpe_ratio = metrics["sharpe_ratio"]
        else:
            sharpe_ratio = 0
        
        # 从回测结果获取累积收益率和最大回撤
        cumulative_returns = metrics["cumulative_returns"]
        annualized_returns = metrics["annualized_returns"]
        max_drawdown = metrics["max_drawdown"]
        
        # 按指定指标返回结果
        if metric == "sharpe_ratio":
            return params, sharpe_ratio
        elif metric == "returns":
            return params, annualized_returns
        elif metric == "drawdown":
            return params, -max_drawdown  # 负号使得可以统一使用最大值作为最优
        else:
            return params, sharpe_ratio
    except Exception as e:
        print(f"评估参数 {params} 时出错: {e}")
        return params, -999  # 错误时返回极低的评分

def optimize_strategy(df, param_ranges=PARAM_RANGES, metric="sharpe_ratio"):
    """
    优化策略参数
    
    参数:
        df: 股票数据DataFrame
        param_ranges: 参数范围字典
        metric: 优化指标，可选 "sharpe_ratio", "returns", "drawdown"
    
    返回:
        results: 包含所有评估结果的列表
        best_params: 最优参数组合
        best_value: 最优指标值
    """
    # 生成所有参数组合
    param_combinations = list(product(
        param_ranges["ma_short"],
        param_ranges["ma_long"],
        param_ranges["rsi_period"],
        param_ranges["rsi_oversold"],
        param_ranges["rsi_overbought"]
    ))
    
    print(f"正在测试 {len(param_combinations)} 个参数组合...")
    
    # 使用多进程并行计算
    num_cores = max(1, mp.cpu_count() - 1)  # 保留一个核心给系统
    print(f"使用 {num_cores} 个CPU核心进行并行计算...")
    
    # 创建部分应用的评估函数
    eval_func = partial(_evaluate_params, df=df, metric=metric)
    
    # 使用进程池并行计算
    with mp.Pool(processes=num_cores) as pool:
        # 使用tqdm显示进度
        results = list(tqdm(pool.imap(eval_func, param_combinations), total=len(param_combinations)))
    
    # 找出最优参数
    best_result = max(results, key=lambda x: x[1])
    best_params = best_result[0]
    best_value = best_result[1]
    
    # 转换为字典格式
    param_names = ["ma_short", "ma_long", "rsi_period", "rsi_oversold", "rsi_overbought"]
    best_params_dict = {name: value for name, value in zip(param_names, best_params)}
    
    # 打印进度
    print(f"优化完成! 最佳参数组合:")
    for name, value in best_params_dict.items():
        print(f"  {name}: {value}")
    
    # 打印最佳指标值
    metric_name = {
        "sharpe_ratio": "夏普比率",
        "returns": "年化收益率(%)",
        "drawdown": "最大回撤(%)"
    }.get(metric, metric)
    
    if metric == "drawdown":
        best_value = -best_value  # 还原回撤的实际值
    
    print(f"最佳{metric_name}: {best_value:.4f}")
    
    return results, best_params_dict, best_value

def plot_optimization_results(results, metric="sharpe_ratio", top_n=20):
    """绘制优化结果的图表"""
    plt.figure(figsize=(15, 8))
    
    # 提取top_n个结果
    top_results = results[:top_n]
    
    # 根据选择的指标获取值
    if metric == "sharpe_ratio":
        values = [r["metrics"]["sharpe_ratio"] for r in top_results]
        title = "夏普比率"
    elif metric == "returns":
        values = [r["metrics"]["annualized_returns"] for r in top_results]
        title = "年化收益率(%)"
    elif metric == "drawdown":
        values = [r["metrics"]["max_drawdown"] for r in top_results]
        title = "最大回撤(%)"
    
    # 创建标签
    labels = [f"MA{r['params']['ma_short']}/{r['params']['ma_long']}, RSI{r['params']['rsi_period']}({r['params']['rsi_oversold']}/{r['params']['rsi_overbought']})" for r in top_results]
    
    # 绘制条形图
    plt.bar(range(len(values)), values)
    plt.xticks(range(len(values)), labels, rotation=90)
    plt.title(f"前{top_n}个参数组合的{title}")
    plt.ylabel(title)
    plt.tight_layout()
    
    # 保存图表
    timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
    filename = f"plots/optimizations/optimization_results_{metric}_{timestamp}.png"
    plt.savefig(filename)
    plt.close()
    
    return filename

def plot_parameter_sensitivity(results, param_name, metric="sharpe_ratio"):
    """分析单个参数对结果的敏感性"""
    plt.figure(figsize=(10, 6))
    
    # 获取参数的所有不同值
    param_values = sorted(list(set([r["params"][param_name] for r in results])))
    
    # 为每个参数值计算平均指标
    avg_metrics = []
    for value in param_values:
        filtered_results = [r for r in results if r["params"][param_name] == value]
        
        if metric == "sharpe_ratio":
            avg_metric = np.mean([r["metrics"]["sharpe_ratio"] for r in filtered_results])
        elif metric == "returns":
            avg_metric = np.mean([r["metrics"]["annualized_returns"] for r in filtered_results])
        elif metric == "drawdown":
            avg_metric = np.mean([r["metrics"]["max_drawdown"] for r in filtered_results])
            
        avg_metrics.append(avg_metric)
    
    # 绘制参数敏感性图
    plt.plot(param_values, avg_metrics, marker='o')
    
    if metric == "sharpe_ratio":
        plt.title(f"参数 {param_name} 对夏普比率的影响")
        plt.ylabel("平均夏普比率")
    elif metric == "returns":
        plt.title(f"参数 {param_name} 对年化收益率的影响")
        plt.ylabel("平均年化收益率(%)")
    elif metric == "drawdown":
        plt.title(f"参数 {param_name} 对最大回撤的影响")
        plt.ylabel("平均最大回撤(%)")
        
    plt.xlabel(param_name)
    plt.grid(True)
    
    # 保存图表
    timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
    filename = f"plots/optimizations/param_sensitivity_{param_name}_{metric}_{timestamp}.png"
    plt.savefig(filename)
    plt.close()
    
    return filename

def apply_best_strategy(df, best_params):
    """应用最佳参数策略到数据上"""
    
    def custom_calculate_indicators(df):
        # 计算移动平均线
        df['MA5'] = df['close'].rolling(window=best_params["ma_short"]).mean()
        df['MA10'] = df['close'].rolling(window=best_params["ma_long"]).mean()
        
        # 计算RSI
        delta = df['close'].diff()
        gain = delta.where(delta > 0, 0)
        loss = -delta.where(delta < 0, 0)
        avg_gain = gain.rolling(window=best_params["rsi_period"]).mean()
        avg_loss = loss.rolling(window=best_params["rsi_period"]).mean()
        rs = avg_gain / avg_loss
        df['RSI'] = 100 - (100 / (1 + rs))
        
        # 计算MACD (使用固定参数)
        ema12 = df['close'].ewm(span=12, adjust=False).mean()
        ema26 = df['close'].ewm(span=26, adjust=False).mean()
        df['MACD'] = ema12 - ema26
        df['Signal_Line'] = df['MACD'].ewm(span=9, adjust=False).mean()
        
        return df
    
    def custom_generate_signals(df):
        df['Signal'] = 0
        
        # 使用移动平均线交叉
        df.loc[(df['MA5'] > df['MA10']) & (df['MA5'].shift(1) <= df['MA10'].shift(1)), 'Signal'] = 1  # 金叉买入
        df.loc[(df['MA5'] < df['MA10']) & (df['MA5'].shift(1) >= df['MA10'].shift(1)), 'Signal'] = -1  # 死叉卖出
        
        # 使用RSI超买超卖
        df.loc[(df['RSI'] < best_params["rsi_oversold"]), 'Signal'] = 1  # RSI超卖买入
        df.loc[(df['RSI'] > best_params["rsi_overbought"]), 'Signal'] = -1  # RSI超买卖出
        
        # 使用MACD交叉
        df.loc[(df['MACD'] > df['Signal_Line']) & (df['MACD'].shift(1) <= df['Signal_Line'].shift(1)), 'Signal'] = 1  # MACD金叉买入
        df.loc[(df['MACD'] < df['Signal_Line']) & (df['MACD'].shift(1) >= df['Signal_Line'].shift(1)), 'Signal'] = -1  # MACD死叉卖出
        
        return df
    
    df = custom_calculate_indicators(df)
    df = custom_generate_signals(df)
    
    df_result, metrics = backtest(df)
    
    return df_result, metrics["cumulative_returns"], metrics["annualized_returns"]

def plot_best_strategy(df, symbol, metrics):
    """绘制最佳策略的回测结果"""
    plt.figure(figsize=(15, 10))
    
    # 子图1: 价格和移动平均线
    plt.subplot(3, 1, 1)
    plt.plot(df['time'], df['close'], label='价格', alpha=0.7)
    
    # 寻找MA列
    ma_columns = [col for col in df.columns if col.startswith('MA') and col[2:].isdigit()]
    for col in ma_columns:
        plt.plot(df['time'], df[col], label=col, alpha=0.7)
    
    # 标记买入和卖出点
    buy_signals = df[df['Trade_Signal'] == 1]
    sell_signals = df[df['Trade_Signal'] == -1]
    plt.scatter(buy_signals['time'], buy_signals['close'], color='green', marker='^', s=100, label='买入信号')
    plt.scatter(sell_signals['time'], sell_signals['close'], color='red', marker='v', s=100, label='卖出信号')
    
    plt.title(f'{symbol} 优化策略 - 价格和交易信号')
    plt.ylabel('价格')
    plt.legend(loc='upper left')
    
    # 子图2: RSI
    plt.subplot(3, 1, 2)
    plt.plot(df['time'], df['RSI'], label='RSI')
    plt.axhline(y=30, color='g', linestyle='-', alpha=0.3)
    plt.axhline(y=70, color='r', linestyle='-', alpha=0.3)
    plt.title('RSI指标')
    plt.ylabel('RSI值')
    plt.legend()
    
    # 子图3: 投资组合价值
    plt.subplot(3, 1, 3)
    plt.plot(df['time'], df['Portfolio'], label='优化策略')
    
    # 计算基准投资组合
    benchmark = df['close'] / df['close'].iloc[0] * 100000
    plt.plot(df['time'], benchmark, label='买入持有策略', alpha=0.7)
    
    plt.title(f'投资组合价值对比 - 收益率: {metrics["cumulative_returns"]:.2f}%, 最大回撤: {metrics["max_drawdown"]:.2f}%')
    plt.ylabel('价值')
    plt.legend()
    
    plt.tight_layout()
    
    # 保存图表
    timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
    filename = f"plots/optimizations/{symbol.replace('.', '_')}_optimized_{timestamp}.png"
    plt.savefig(filename)
    plt.close()
    
    return filename

def main(stock_data_file, symbol, metric="sharpe_ratio", use_limit=True):
    """主函数，运行策略优化流程"""
    # 创建结果文件夹
    os.makedirs(f"results/optimizations/{symbol}", exist_ok=True)
    
    # 加载股票数据
    df = pd.read_csv(stock_data_file)
    
    # 限制数据量以加速处理（仅用于开发测试）
    if use_limit and len(df) > 100:
        print(f"限制数据量为最近100条记录以加速处理")
        df = df.tail(100).reset_index(drop=True)
    
    print(f"已加载 {symbol} 的股票数据，共 {len(df)} 行")
    
    # 优化策略参数
    print("开始优化策略参数...")
    results, best_params, best_value = optimize_strategy(df, metric=metric)
    
    # 保存优化结果
    timestamp = datetime.now().strftime("%Y%m%d%H%M%S")
    result_file = f"results/optimizations/{symbol}/optimization_{timestamp}.json"
    
    results_to_save = {
        "symbol": symbol,
        "timestamp": timestamp,
        "metric": metric,
        "best_params": best_params,
        "best_value": best_value,
        "top_results": [{
            "params": {
                "ma_short": r[0][0], 
                "ma_long": r[0][1], 
                "rsi_period": r[0][2], 
                "rsi_oversold": r[0][3], 
                "rsi_overbought": r[0][4]
            },
            "value": float(r[1])
        } for r in sorted(results, key=lambda x: x[1], reverse=True)[:20]]
    }
    
    with open(result_file, 'w') as f:
        json.dump(results_to_save, f, indent=4)
    
    print(f"优化结果已保存到 {result_file}")
    
    # 应用最佳参数和绘制图表
    df_result, cumulative_returns, annualized_returns = apply_best_strategy(df, best_params)
    
    # 绘制和保存结果图表
    metrics_for_plot = {
        "cumulative_returns": cumulative_returns,
        "annualized_returns": annualized_returns,
        "max_drawdown": 0  # 由于我们使用自定义回测函数不返回最大回撤，这里设为0
    }
    
    plot_best_strategy(df_result, symbol, metrics_for_plot)
    
    return best_params, df_result

if __name__ == "__main__":
    import sys
    
    # 命令行参数处理
    if len(sys.argv) < 3:
        print("用法: python strategy_optimizer.py <股票数据文件> <股票代码> [优化指标] [快速模式]")
        print("优化指标可选: sharpe_ratio, returns, drawdown")
        print("快速模式可选: true/false，默认为true")
        sys.exit(1)
    
    stock_data_file = sys.argv[1]
    symbol = sys.argv[2]
    metric = sys.argv[3] if len(sys.argv) > 3 else "sharpe_ratio"
    use_limit = True if len(sys.argv) <= 4 else sys.argv[4].lower() == "true"
    
    print(f"运行优化 - 股票数据: {stock_data_file}, 股票代码: {symbol}, 指标: {metric}, 快速模式: {use_limit}")
    main(stock_data_file, symbol, metric, use_limit) 