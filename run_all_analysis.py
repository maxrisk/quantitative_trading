#!/usr/bin/env python
# -*- coding: utf-8 -*-

import os
import sys
import asyncio
import subprocess
import time
from datetime import datetime
from dotenv import load_dotenv

# 加载环境变量
load_dotenv()

# 获取命令行参数
auto_trade = "--auto-trade" in sys.argv
only_auto_trade = "--only-auto-trade" in sys.argv
target_symbol = None

# 查找--symbol参数
for i, arg in enumerate(sys.argv):
    if arg == "--symbol" and i + 1 < len(sys.argv):
        target_symbol = sys.argv[i + 1]
        break

# 检查API密钥是否已设置
def check_api_keys():
    required_keys = ["LONGPORT_APP_KEY", "LONGPORT_APP_SECRET", "LONGPORT_ACCESS_TOKEN", "DEEPSEEK_API_KEY"]
    missing_keys = [key for key in required_keys if not os.getenv(key)]
    
    if missing_keys:
        print(f"错误: 缺少以下环境变量: {', '.join(missing_keys)}")
        print("请确保在.env文件中设置了所有必要的API密钥")
        return False
    
    return True

# 创建结果目录
def create_directories():
    directories = ["results", "plots", "results/optimizations", "plots/optimizations"]
    for directory in directories:
        os.makedirs(directory, exist_ok=True)
    print("已创建结果目录")

# 运行基本分析
async def run_basic_analysis():
    print("\n" + "="*50)
    print("运行基本分析 (quant_trading.py)")
    print("="*50)
    
    # 导入基本分析模块并运行
    if target_symbol:
        print(f"专注分析股票: {target_symbol}")
        # 使用subprocess调用，传递股票代码作为参数
        try:
            command = [sys.executable, "quant_trading.py", target_symbol]
            process = subprocess.Popen(command, stdout=subprocess.PIPE, stderr=subprocess.PIPE, universal_newlines=True)
            
            # 实时输出结果
            for line in process.stdout:
                print(line.strip())
            
            process.wait()
            
            if process.returncode != 0:
                print("基本分析执行失败")
                for line in process.stderr:
                    print(line.strip())
                return False
            return True
        except Exception as e:
            print(f"运行基本分析时发生错误: {e}")
            return False
    else:
        # 原有的分析方式
        from quant_trading import main
        await main()
        return True

# 运行多股票分析
async def run_multi_stock_analysis():
    print("\n" + "="*50)
    print("运行多股票分析 (multi_stock_analysis.py)")
    print("="*50)
    
    # 导入多股票分析模块并运行
    from multi_stock_analysis import main
    await main()

# 运行策略优化
def run_strategy_optimization(fast_mode=True):
    print("\n" + "="*50)
    print("运行策略优化 (strategy_optimizer.py)")
    print("="*50)
    
    # 检查是否有股票数据文件
    result_files = [f for f in os.listdir("results") if f.endswith("_analysis.csv")]
    
    if not result_files:
        print("错误: 没有找到可用于优化的股票数据文件")
        print("请先运行基本分析或多股票分析生成数据文件")
        return
    
    # 选择第一个股票文件进行优化
    data_file = os.path.join("results", result_files[0])
    symbol = result_files[0].replace("_analysis.csv", "").replace("_", ".")
    
    print(f"为 {symbol} 运行策略优化...")
    
    # 使用subprocess调用策略优化脚本
    try:
        # 使用新命令行参数格式
        command = [
            sys.executable, 
            "strategy_optimizer.py",
            data_file,
            symbol,
            "sharpe_ratio",
            "true" if fast_mode else "false"
        ]
        
        print(f"执行命令: {' '.join(command)}")
        
        process = subprocess.Popen(
            command, 
            stdout=subprocess.PIPE, 
            stderr=subprocess.PIPE, 
            universal_newlines=True
        )
        
        # 实时输出结果
        for line in process.stdout:
            print(line.strip())
        
        process.wait()
        
        if process.returncode != 0:
            print("策略优化执行失败")
            for line in process.stderr:
                print(line.strip())
    except Exception as e:
        print(f"运行策略优化时发生错误: {e}")

# 运行自动交易
async def run_auto_trading(auto_trade=False):
    if not auto_trade:
        print("\n" + "="*50)
        print("自动交易未启用，跳过")
        print("要启用自动交易，请使用 --auto-trade 参数")
        print("="*50)
        return
    
    print("\n" + "="*50)
    print("运行自动交易 (auto_trader.py)")
    if target_symbol:
        print(f"专注交易股票: {target_symbol}")
    print("="*50)
    
    # 导入自动交易模块并运行
    from auto_trader import auto_trading_loop
    await auto_trading_loop(target_symbol)

# 生成汇总报告
def generate_summary_report():
    print("\n" + "="*50)
    print("生成汇总报告")
    print("="*50)
    
    # 收集结果文件
    result_files = [f for f in os.listdir("results") if f.endswith("_analysis.csv")]
    optimization_files = [f for f in os.listdir("results/optimizations") if f.endswith(".json")]
    portfolio_file = "results/portfolio_allocation.csv"
    
    # 创建报告文件
    timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
    report_file = f"results/summary_report_{timestamp}.md"
    
    with open(report_file, "w") as f:
        f.write("# 量化交易分析汇总报告\n\n")
        f.write(f"生成时间: {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}\n\n")
        
        # 添加分析的股票
        f.write("## 分析的股票\n\n")
        for file in result_files:
            symbol = file.replace("_analysis.csv", "").replace("_", ".")
            f.write(f"- {symbol}\n")
        
        # 投资组合分析
        f.write("\n## 投资组合分析\n\n")
        if os.path.exists(portfolio_file):
            import pandas as pd
            try:
                portfolio_data = pd.read_csv(portfolio_file)
                f.write("### 推荐投资组合权重\n\n")
                f.write("| 股票 | 权重 | 预期年化收益率 | 预期年化波动率 |\n")
                f.write("|------|------|----------------|----------------|\n")
                
                for _, row in portfolio_data.iterrows():
                    f.write(f"| {row['Symbol']} | {row['Weight']:.2f}% | {row['Expected Annual Return']:.2f}% | {row['Expected Annual Volatility']:.2f}% |\n")
            except Exception as e:
                f.write(f"读取投资组合数据时发生错误: {e}\n")
        else:
            f.write("未找到投资组合分析数据\n")
        
        # 策略优化结果
        f.write("\n## 策略优化结果\n\n")
        if optimization_files:
            import json
            for file in optimization_files:
                try:
                    with open(os.path.join("results/optimizations", file), "r") as opt_file:
                        opt_data = json.load(opt_file)
                        
                    f.write(f"### {opt_data.get('symbol', '未知股票')} 优化结果\n\n")
                    
                    # 最佳参数
                    f.write("#### 最佳参数\n\n")
                    f.write("| 参数 | 值 |\n")
                    f.write("|------|----|\n")
                    for param, value in opt_data.get("best_params", {}).items():
                        f.write(f"| {param} | {value} |\n")
                    
                    # 最佳绩效
                    f.write("\n#### 最佳绩效\n\n")
                    f.write("| 指标 | 值 |\n")
                    f.write("|------|----|\n")
                    metrics = opt_data.get("best_metrics", {})
                    f.write(f"| 累计收益率 | {metrics.get('cumulative_returns', 0):.2f}% |\n")
                    f.write(f"| 年化收益率 | {metrics.get('annualized_returns', 0):.2f}% |\n")
                    f.write(f"| 最大回撤 | {metrics.get('max_drawdown', 0):.2f}% |\n")
                    f.write(f"| 夏普比率 | {metrics.get('sharpe_ratio', 0):.2f} |\n")
                    
                except Exception as e:
                    f.write(f"读取优化结果时发生错误: {e}\n")
        else:
            f.write("未找到策略优化结果\n")
        
        # 交易状态信息
        f.write("\n## 交易状态\n\n")
        trading_state_file = "results/trading_state.json"
        if os.path.exists(trading_state_file):
            import json
            try:
                with open(trading_state_file, "r") as state_file:
                    state = json.load(state_file)
                
                f.write(f"最后更新时间: {state.get('timestamp', '未知')}\n\n")
                
                f.write("### 当前持仓\n\n")
                positions = state.get("positions", {})
                if positions:
                    f.write("| 股票 | 数量 | 成本价 | 当前价 | 市值 | 盈亏比例 |\n")
                    f.write("|------|------|--------|--------|------|----------|\n")
                    
                    for symbol, pos in positions.items():
                        cost_price = pos.get("cost_price", 0)
                        current_price = pos.get("current_price", 0)
                        profit_pct = ((current_price / cost_price) - 1) * 100 if cost_price else 0
                        
                        f.write(f"| {symbol} | {pos.get('quantity', 0)} | {cost_price:.2f} | {current_price:.2f} | {pos.get('market_value', 0):.2f} | {profit_pct:.2f}% |\n")
                else:
                    f.write("当前无持仓\n")
                
                f.write("\n### 最近订单\n\n")
                orders = state.get("orders", {})
                if orders:
                    f.write("| 订单ID | 股票 | 方向 | 数量 | 价格 | 状态 | 时间 |\n")
                    f.write("|--------|------|------|------|------|------|------|\n")
                    
                    # 取最近10个订单
                    recent_orders = list(orders.items())[-10:]
                    for order_id, order in recent_orders:
                        f.write(f"| {order_id} | {order.get('symbol', '')} | {order.get('side', '')} | {order.get('quantity', 0)} | {order.get('price', 0):.2f} | {order.get('status', '')} | {order.get('time', '')} |\n")
                else:
                    f.write("无最近订单记录\n")
                    
            except Exception as e:
                f.write(f"读取交易状态数据时发生错误: {e}\n")
        else:
            f.write("未找到交易状态数据\n")
        
        # 图表链接
        f.write("\n## 生成的图表\n\n")
        plot_files = [f for f in os.listdir("plots") if f.endswith(".png")]
        plot_files += [os.path.join("optimizations", f) for f in os.listdir("plots/optimizations") if f.endswith(".png")]
        
        if plot_files:
            for plot in plot_files:
                plot_path = os.path.join("plots", plot)
                f.write(f"- [{plot}]({plot_path})\n")
        else:
            f.write("未找到图表文件\n")
    
    print(f"汇总报告已生成: {report_file}")
    return report_file

# 主函数
async def main():
    print("="*50)
    print("量化交易分析系统")
    print("="*50)
    
    start_time = time.time()
    
    # 检查API密钥
    if not check_api_keys():
        return
    
    # 创建必要的目录
    create_directories()
    
    # 检查是否有快速模式参数
    fast_mode = "--fast" in sys.argv
    if fast_mode:
        print("快速模式已启用，将使用有限数据进行优化")
    else:
        print("标准模式运行，将使用全部数据进行优化")
    
    if only_auto_trade:
        # 仅运行自动交易
        await run_auto_trading(True)
    else:
        # 运行基本分析
        basic_analysis_success = await run_basic_analysis()
        
        if not target_symbol:
            # 只有在未指定目标股票时才运行多股票分析
            await run_multi_stock_analysis()
            
            # 运行策略优化
            run_strategy_optimization(fast_mode)
        
        # 运行自动交易（如果启用）
        await run_auto_trading(auto_trade)
    
    # 生成汇总报告
    report_file = generate_summary_report()
    
    # 计算总耗时
    elapsed_time = time.time() - start_time
    print("\n" + "="*50)
    print(f"分析完成，总耗时: {elapsed_time:.2f} 秒")
    print(f"汇总报告: {report_file}")
    print("="*50)

if __name__ == "__main__":
    asyncio.run(main()) 