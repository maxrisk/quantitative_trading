# 量化交易程序

基于长桥API和DeepSeek AI的量化交易分析工具，使用pandas进行技术指标分析。

## 功能特点

- 获取实时股票数据
- 计算常用技术指标（MA, RSI, MACD, 布林带等）
- 生成交易信号和策略
- 进行历史数据回测
- 使用DeepSeek AI进行市场分析和预测
- 数据结果导出为CSV文件
- **多股票组合分析和投资组合优化**
- **策略参数优化和敏感性分析**
- **自动化分析流程和汇总报告生成**
- **自动化交易执行（模拟交易和实盘交易）**

## 环境要求

- Python 3.8+
- pandas
- numpy
- matplotlib
- requests
- python-dotenv
- longport
- tqdm

## 安装

1. 克隆代码库
```bash
git clone <repository-url>
cd quantitative_trading
```

2. 安装依赖包
```bash
pip install pandas numpy matplotlib longport python-dotenv requests tqdm
```

3. 配置API密钥
在项目根目录创建`.env`文件，并添加以下内容：
```
LONGPORT_APP_KEY="your_app_key"
LONGPORT_APP_SECRET="your_app_secret" 
LONGPORT_ACCESS_TOKEN="your_access_token"
DEEPSEEK_API_KEY="your_deepseek_api_key"
```

## 使用方法

### 一键运行所有分析

运行所有分析流程并生成汇总报告：
```bash
python run_all_analysis.py
```

这将依次执行：
1. 单股票基本分析
2. 多股票组合分析
3. 策略参数优化
4. 生成汇总报告

### 运行分析和自动交易

运行分析并启动自动交易功能：
```bash
python run_all_analysis.py --auto-trade
```

### 仅运行自动交易

仅启动自动交易功能（不运行分析）：
```bash
python run_all_analysis.py --only-auto-trade
```

### 单独运行自动交易

也可以单独运行自动交易脚本：
```bash
python auto_trader.py
```

### 基本分析

运行主程序分析单个股票：
```bash
python quant_trading.py
```

### 多股票组合分析

运行多股票分析程序：
```bash
python multi_stock_analysis.py
```

这将分析预设的多只股票，并提供组合投资建议。

### 策略参数优化

运行策略优化程序：
```bash
python strategy_optimizer.py --file results/700_HK_analysis.csv --symbol 700.HK --metric sharpe_ratio
```

参数说明：
- `--file`: 股票数据CSV文件路径
- `--symbol`: 股票代码
- `--metric`: 优化指标，可选 "sharpe_ratio"(夏普比率), "returns"(收益率), "drawdown"(最大回撤)

## 程序说明

### 自动交易 (auto_trader.py)

自动交易程序执行以下功能：
1. 读取分析结果中的交易信号
2. 根据信号自动执行买入/卖出操作
3. 管理持仓和订单状态
4. 实现止损和止盈功能
5. 支持模拟交易和实盘交易两种模式
6. 保存和恢复交易状态

自动交易配置可在脚本中的`TRADE_CONFIG`字典中修改：
```python
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
```

### 一键分析 (run_all_analysis.py)

全流程分析程序执行以下步骤：
1. 检查环境和API密钥配置
2. 创建必要的目录结构
3. 运行单股票基本分析
4. 运行多股票组合分析
5. 运行策略参数优化
6. 收集所有结果并生成汇总报告（Markdown格式）
7. 可选：启动自动交易功能

### 基本分析 (quant_trading.py)

主程序执行以下步骤：
1. 获取指定股票的K线数据
2. 计算技术指标
3. 生成交易信号
4. 进行策略回测
5. 通过AI分析市场走势
6. 保存分析结果到CSV文件

### 多股票分析 (multi_stock_analysis.py)

多股票分析程序执行以下步骤：
1. 获取多只股票的K线数据
2. 为每只股票计算技术指标和交易信号
3. 对每只股票进行回测
4. 生成每只股票的AI分析报告
5. 计算股票相关性矩阵
6. 基于夏普比率优化投资组合权重
7. 保存结果到CSV文件

### 策略优化 (strategy_optimizer.py)

策略优化程序执行以下步骤：
1. 读取股票数据
2. 在多个参数组合上测试策略表现
3. 找出最优参数组合
4. 绘制优化结果图表
5. 分析参数敏感性
6. 使用最优参数重新回测
7. 保存优化结果和图表

## 交易信号

本程序生成的交易信号基于以下策略：

- MA5与MA10的黄金交叉（买入）和死亡交叉（卖出）
- RSI超买（>70，卖出）和超卖（<30，买入）信号
- MACD与信号线的交叉

## 自动交易功能

自动交易功能具有以下特点：

- 自动处理交易信号并执行交易
- 支持模拟交易和实盘交易模式
- 实现止损和止盈功能
- 按整手（100股）计算买入数量
- 根据配置控制仓位比例和风险
- 交易状态持久化，支持程序重启后恢复

## 投资组合分析

投资组合分析具有以下功能：

- 计算股票间的相关性矩阵
- 基于夏普比率优化投资组合权重
- 计算组合的预期收益率和风险
- 生成投资建议

## 策略优化功能

策略优化工具支持：

- 对多个策略参数进行网格搜索
- 针对不同指标（夏普比率、收益率、最大回撤）的优化
- 参数敏感性分析
- 生成优化结果可视化图表
- 保存优化后的策略参数和回测数据

## 输出结果

程序会输出以下结果：
- 回测结果（累计收益率、年化收益率、最大回撤、夏普比率）
- 最近交易信号
- AI市场分析报告
- CSV文件（包含所有技术指标和交易信号）
- 投资组合分析结果
- 策略优化结果和图表
- Markdown格式的汇总分析报告
- 交易状态和执行记录

## 项目结构

```
quantitative_trading/
├── .env                      # API密钥配置文件
├── quant_trading.py          # 基本分析程序
├── multi_stock_analysis.py   # 多股票分析程序
├── strategy_optimizer.py     # 策略优化程序
├── auto_trader.py            # 自动交易程序
├── run_all_analysis.py       # 一键运行所有分析
├── README.md                 # 项目说明文档
├── results/                  # 分析结果目录
│   ├── portfolio_allocation.csv  # 投资组合配置
│   ├── *_analysis.csv        # 各股票分析结果
│   ├── summary_report_*.md   # 汇总报告
│   ├── trading_state.json    # 交易状态记录
│   └── optimizations/        # 优化结果
├── plots/                    # 图表目录
│   └── optimizations/        # 优化图表
```

## 示例输出

```
正在获取 700.HK 的K线数据...
计算技术指标...
生成交易信号...
进行回测...

回测结果:
累计收益率: 15.23%
年化收益率: 45.67%
最大回撤: 8.75%
夏普比率: 1.92

最近交易信号:
                  time    close  Signal
2023-06-01  2023-06-01T00:00:00  380.40      0
2023-06-02  2023-06-02T00:00:00  382.60      1
2023-06-03  2023-06-03T00:00:00  379.80     -1
2023-06-04  2023-06-04T00:00:00  385.20      0
2023-06-05  2023-06-05T00:00:00  390.40      1

通过AI进行市场分析...
[AI分析结果将显示在这里]

分析结果已保存到 700_HK_analysis.csv
```

## 注意事项

- 本程序仅供学习和研究使用，不构成投资建议
- 交易决策请自行判断，使用真实资金交易前请充分测试策略
- API密钥信息请妥善保管，不要泄露给他人
- 策略优化可能需要较长时间，请耐心等待
- 投资组合建议仅作参考，实际投资决策应考虑更多因素
- 默认为模拟交易模式，切换到实盘交易前请务必谨慎评估风险 