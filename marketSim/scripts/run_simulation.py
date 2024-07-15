import logging
import random
from marketSim.utils.io import load_config, setup_logging
from marketSim.main import Wallet


def run_simulation():
    # Load configuration
    config = load_config('config/config.yaml')
    setup_logging(config)

    logging.info("Starting simulation")

    # Initialize Wallet
    wallet = Wallet(
        start_currency=config['simulation']['start_currency'],
        start_val=config['simulation']['start_val'],
        episode_len=config['simulation']['episode_len'],
        pred_len=config['simulation']['pred_len'],
        interval=config['simulation']['interval'],
        available_markets=config['markets']
    )

    # Reset environment
    state = wallet.reset()

    done = False
    while not done:
        actions = [random.uniform(-1, 1) for _ in range(wallet.action_space)]
        state, reward, done = wallet.step(actions)
        logging.info(f"Actions: {actions}, Reward: {reward}")

    logging.info("Simulation ended")
    wallet.print_wallet()


if __name__ == "__main__":
    run_simulation()
