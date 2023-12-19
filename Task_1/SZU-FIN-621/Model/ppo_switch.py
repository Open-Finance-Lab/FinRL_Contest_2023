# -*- coding: utf-8 -*-
"""
@author: Zhong Anyang, Chao Kaiyin, and Chen Geying from Shenzhen University
@supervisor: Yin Jianfei and Joshua Zhexue Huang
@describe: This software serves as our submission for the 4th ACM ICAIF 2023 FinRL Contest.
@date: 2023-11-12
"""

import numpy as np


class PPO_Switch:
    def __init__(self, stocksDimension, switchWindows, hmax=100, alpha=0.5):
        self.switchWindows = switchWindows
        self.stocksDim = stocksDimension
        self.alpha = alpha
        self.hamx =hmax
        pass

    def DRL_prediction(self, model, environment, deterministic=True):
        """
            make a prediction and get results
            :param model: (list<object>) multiple different model
            :param environment: (list<object>) a final test environment and multiple models corresponding environment
            :param deterministic: (bool) Whether or not to return deterministic actions.
            :return: (df) cumulative wealth and actions record in different periods
        """

        ppo_switch_env, ppo_switch_obs = environment[0].get_sb_env()
        ppo_real_env, ppo_real_obs = environment[1].get_sb_env()
        ppo_max_env, ppo_max_obs = environment[2].get_sb_env()
        ppo_min_env, ppo_min_obs = environment[3].get_sb_env()
        ppo_mean_env, ppo_mean_obs = environment[4].get_sb_env()
        ppo_ema_env, ppo_ema_obs = environment[5].get_sb_env()

        account_memory = None  # This help avoid unnecessary list creation
        actions_memory = None  # optimize memory consumption

        ppo_switch_env.reset()
        ppo_real_env.reset()
        ppo_max_env.reset()
        ppo_min_env.reset()
        ppo_mean_env.reset()
        ppo_ema_env.reset()

        max_steps = len(environment[0].df.index.unique()) - 1
        switchWindow = self.switchWindows[0]
        lastSwitchCW = [0 for _ in range(len(model))]
        lastSwitchModelIndex = 0
        switchFactor = [0 for _ in range(len(model))]

        for i in range(len(environment[0].df.index.unique())):
            ppo_real_action, _realStates = model[0].predict(ppo_real_obs, deterministic=deterministic)
            ppo_max_action, _maxStates = model[1].predict(ppo_max_obs, deterministic=deterministic)
            ppo_min_action, _minStates = model[2].predict(ppo_min_obs, deterministic=deterministic)
            ppo_mean_action, _meanStates = model[3].predict(ppo_mean_obs, deterministic=deterministic)
            ppo_ema_action, _emaStates = model[4].predict(ppo_ema_obs, deterministic=deterministic)

            all_actions = [
                ppo_real_action,
                ppo_max_action,
                ppo_min_action,
                ppo_mean_action,
                ppo_ema_action,
            ]

            # Init final action
            finalAction = all_actions[lastSwitchModelIndex]

            # When the number of days is greater than or equal to 10 days, execute the sparsification and switching rules
            if i >= 10:
                # Obtain the rewards for different models from day (i-5) to day i.
                rewards = self.get_modelReward(environment, i, 5)

                # Obtain the cumulative wealth of different models on day i.
                newCW = self.get_modelNowCW(environment, i)

                # Calculate the switchFactor of different models by the short-term average rewards and long-term rewards
                for j in range(len(switchFactor)):
                    switchFactor[j] = self.alpha * rewards[j] / 5 + (1 - self.alpha) * (newCW[j] - lastSwitchCW[j])

                # Find the index of the maximum value in the switchFactor list
                nowSwitchModelIndex = switchFactor.index(max(switchFactor))
                if i % switchWindow == 0:
                    if nowSwitchModelIndex != lastSwitchModelIndex:
                        # Record new selected model action index
                        lastSwitchModelIndex = nowSwitchModelIndex
                        lastSwitchCW = newCW
                    chooseAction = all_actions[nowSwitchModelIndex]

                    # Sparse the chooseAction (invest in only one stock)
                    finalAction = self.sparse_action(environment[0], chooseAction, self.stocksDim)

                    # According to the reward size within the window time, select the maximum reward window time
                    # in the self.switchWindows as the next switching window
                    cwList = [0 for _ in range(len(self.switchWindows))]
                    for j in range(len(self.switchWindows)):
                        preCW = self.get_modelReward(environment, i, self.switchWindows[j])
                        for k in range(len(newCW)):
                            cwList[j] += preCW[k]
                    switchWindow = self.switchWindows[np.argmax(cwList)]
                else:
                    # hold action
                    finalAction = [np.array([0 for _ in range(self.stocksDim)])]

            ppo_max_obs, _, _, _ = ppo_max_env.step(ppo_max_action)
            ppo_min_obs, _, _, _ = ppo_min_env.step(ppo_min_action)
            ppo_mean_obs, _, _, _ = ppo_mean_env.step(ppo_mean_action)
            ppo_ema_obs, _, _, _ = ppo_ema_env.step(ppo_ema_action)
            ppo_real_obs, _, _, _ = ppo_real_env.step(ppo_real_action)

            _, _, dones, _ = ppo_switch_env.step(finalAction)

            if i == max_steps - 1:  # more descriptive condition for early termination to clarify the logic
                account_memory = ppo_switch_env.env_method(method_name="save_asset_memory")
                actions_memory = ppo_switch_env.env_method(method_name="save_action_memory")
            # add current state to state memory
            # state_memory=test_env.env_method(method_name="save_state_memory")
            if dones[0]:
                print("hit end!")
                break
        return account_memory[0], actions_memory[0]

    def sparse_action(self, env, action, stocks_num):
        """
        Sparse action function that generates a sparse action based on the given environment, action, and number of stocks.

        :param env: (object) Environment object
        :param action: (array[0][stocks_num]) Action array
        :param stocks_num: (int) Number of stocks
        :return: (array[0][stocks_num]) Sparse action array
        """
        # Get the currently held quantity of stocks in the env
        heldQuantity = env.state[stocks_num + 1:2 * stocks_num + 1]
        newAction = np.array(heldQuantity)/-self.hamx

        # Find the index of the maximum value in the action array
        maxActionIndex = np.argmax(action[0])
        if action[0][int(maxActionIndex)] >= 0:
            newAction[int(maxActionIndex)] = 1000000.0
        return [newAction]

    def get_modelNowCW(self, environments, day):
        """
        Retrieves the current cumulative wealth from the given environments for a specific day.

        :param environments: (list<object>) List of environment objects
        :param day: (int) Day index
        :return: (list<int>) List of cumulative wealth for the given day
        """
        res = [0 for _ in range(len(environments) - 1)]
        # Iterate over the environments (except the first one)
        for i in range(len(environments) - 1):
            res[i] = environments[i + 1].asset_memory[day]
        return res

    def get_modelReward(self, environments, day, interval):
        """
        Calculates the reward based on the change in cumulative wealth between the current day and a specified interval.

        :param environments: (list) List of environment objects
        :param day: Current day index
        :param interval: Time interval
        :return: List of rewards for the given interval
        """
        res = [0 for _ in range(len(environments) - 1)]
        for i in range(len(environments) - 1):
            res[i] = environments[i + 1].asset_memory[day] - environments[i + 1].asset_memory[day - interval]
        return res
