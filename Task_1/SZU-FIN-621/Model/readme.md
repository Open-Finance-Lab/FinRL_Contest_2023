[//]: # (# PPO Switch)

[//]: # ()
[//]: # (A brief description of what the project does.)

## Getting Started

Follow these steps to get started with the PPO Switch project:

### Prerequisites
- Python 3.10

### Installation
1. Install dependencies:
    ```bash
    pip install -r requirements.txt
    ```
2. Install FinRL for Windows:
   ```bash
    git clone https://github.com/AI4Finance-Foundation/FinRL.git
    cd FinRL
    pip install .
    ```

### Running the Train
1. Open the `train.py` script and locate the following lines:
    ```python
   TRAIN_START_DATE = 'YOUR_TRAIN_START_DATE'
   TRAIN_END_DATE = 'YOUR_TRAIN_END_DATE'
   model_name = 'YOUR_TRAIN_MODEL_NAME'
   processed_full = 'YOUR_TRAIN_DATA_FILE_PATH'
   ```
   Replace the placeholders with the actual information of your train data file.

2. Run the test using the following command:
    ```bash
    python train.py
    ```

### Running the Test
1. Open the `test.py` script and locate the following lines:
    ```python
   TRADE_START_DATE = 'YOUR_TRADE_START_DATE'
   TRADE_END_DATE = 'YOUR_TRADE_END_DATE'
   FILE_PATH = 'YOUR_TEST_DATA_FILE_PATH'
   ```
   Replace the placeholders with the actual information of your test data file.

   The models, namely ppo_real, ppo_max, ppo_min, ppo_mean, and ppo_ema, are trained on the following datasets: real_train_data.csv, max_train_data.csv, min_train_data.csv, mean_train_data.csv, and ema_train_data.csv, respectively.

2. Run the test using the following command:
    ```bash
    python test.py
    ```