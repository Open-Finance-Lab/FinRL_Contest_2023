"""
@author: Zhong Anyang, Chao Kaiyin, and Chen Geying from Shenzhen University
@supervisor: Yin Jianfei and Joshua Zhexue Huang
@describe: This software serves as our submission for the 4th ACM ICAIF 2023 FinRL Contest.
@date: 2023-11-12
"""
import pandas as pd
import matplotlib.pyplot as plt
import argparse

from finrl.meta.preprocessor.preprocessors import data_split
from finrl.meta.env_stock_trading.env_stocktrading import StockTradingEnv
from finrl.agents.stablebaselines3.models import DRLAgent
from stable_baselines3 import PPO
from finrl.main import check_and_make_directories
from finrl.config import INDICATORS, TRAINED_MODEL_DIR
from finrl.plot import backtest_stats

# Multiple agents switching model designed by us
from ppo_switch import PPO_Switch

# Contestants are welcome to split the data in their own way for model tuning

TRADE_START_DATE = '2021-01-01'
TRADE_END_DATE = '2023-06-30'
FILE_PATH = 'train_data.csv'

# PPO configs
PPO_PARAMS = {
    "n_steps": 2048,
    "ent_coef": 0.01,
    "learning_rate": 0.0003,
    "batch_size": 128,
}

if __name__ == '__main__':
    # We will use unseen, post-deadline data for testing
    parser = argparse.ArgumentParser(description='Description of program')
    parser.add_argument('--start_date', default=TRADE_START_DATE,
                        help='Trade start date (default: {})'.format(TRADE_START_DATE))
    parser.add_argument('--end_date', default=TRADE_END_DATE,
                        help='Trade end date (default: {})'.format(TRADE_END_DATE))
    parser.add_argument('--data_file', default=FILE_PATH, help='Trade data file')

    args = parser.parse_args()
    TRADE_START_DATE = args.start_date
    TRADE_END_DATE = args.end_date

    processed_full = pd.read_csv(args.data_file)
    trade = data_split(processed_full, TRADE_START_DATE, TRADE_END_DATE)

    stock_dimension = len(trade.tic.unique())
    state_space = 1 + 2 * stock_dimension + len(INDICATORS) * stock_dimension
    print(f"Stock Dimension: {stock_dimension}, State Space: {state_space}")

    buy_cost_list = sell_cost_list = [0.001] * stock_dimension
    num_stock_shares = [0] * stock_dimension

    # please do not change initial_amount
    env_kwargs = {
        "hmax": 100,
        "initial_amount": 1000000,
        "num_stock_shares": num_stock_shares,
        "buy_cost_pct": buy_cost_list,
        "sell_cost_pct": sell_cost_list,
        "state_space": state_space,
        "stock_dim": stock_dimension,
        "tech_indicator_list": INDICATORS,
        "action_space": stock_dimension,
        "reward_scaling": 1e-4
    }

    check_and_make_directories([TRAINED_MODEL_DIR])

    # Environment
    e_trade_gym_switch = StockTradingEnv(df=trade, **env_kwargs)
    e_trade_gym_real = StockTradingEnv(df=trade, **env_kwargs)
    e_trade_gym_max = StockTradingEnv(df=trade, **env_kwargs)
    e_trade_gym_min = StockTradingEnv(df=trade, **env_kwargs)
    e_trade_gym_mean = StockTradingEnv(df=trade, **env_kwargs)
    e_trade_gym_ema = StockTradingEnv(df=trade, **env_kwargs)

    # PPO agent
    # ppo_real model is trained with the real train datas
    agent_real = DRLAgent(env=e_trade_gym_real)
    model_ppo_real = agent_real.get_model("ppo", model_kwargs=PPO_PARAMS)
    ppo_real = PPO.load(TRAINED_MODEL_DIR + '/ppo_real')

    # ppo_max, ppo_min, ppo_mean and ppo_ema models are trained using four fake training datas respectively,
    # which is generated based on real data
    agent_max = DRLAgent(env=e_trade_gym_max)
    model_ppo_max = agent_max.get_model("ppo", model_kwargs=PPO_PARAMS)
    ppo_max = PPO.load(TRAINED_MODEL_DIR + '/ppo_max')

    agent_min = DRLAgent(env=e_trade_gym_min)
    model_ppo_min = agent_min.get_model("ppo", model_kwargs=PPO_PARAMS)
    ppo_min = PPO.load(TRAINED_MODEL_DIR + '/ppo_min')

    agent_mean = DRLAgent(env=e_trade_gym_switch)
    model_ppo_mean = agent_mean.get_model("ppo", model_kwargs=PPO_PARAMS)
    ppo_mean = PPO.load(TRAINED_MODEL_DIR + '/ppo_mean')

    agent_ema = DRLAgent(env=e_trade_gym_ema)
    model_ppo_ema = agent_ema.get_model("ppo", model_kwargs=PPO_PARAMS)
    ppo_ema = PPO.load(TRAINED_MODEL_DIR + '/ppo_ema')

    # Backtesting
    ppo_switch = PPO_Switch(alpha=0.3, switchWindows=[2, 5, 7], hmax=env_kwargs['hmax'], stocksDimension=stock_dimension)
    df_result_ppo, df_actions_ppo = ppo_switch.DRL_prediction(
        model=[ppo_real, ppo_max, ppo_min, ppo_mean, ppo_ema],
        environment=[e_trade_gym_switch, e_trade_gym_real, e_trade_gym_max,
                     e_trade_gym_min, e_trade_gym_mean, e_trade_gym_ema])

    print("==============Get Backtest Results===========")
    perf_stats_all = backtest_stats(account_value=df_result_ppo)

    """Plotting"""
    plt.rcParams["figure.figsize"] = (15, 5)
    plt.figure()

    df_result_ppo.plot()
    plt.savefig("plot.png")

    df_result_ppo.to_csv("results.csv", index=False)
