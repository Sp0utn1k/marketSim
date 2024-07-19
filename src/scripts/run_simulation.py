import logging
import random
from src.utils.io import load_config, setup_logging
from src.main import Wallet

import cProfile
import pstats
import io


def run_simulation():
    # Load configuration
    config = load_config('src/config/config.yaml')
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
    profiler = cProfile.Profile()
    profiler.enable()

    run_simulation()

    profiler.disable()
    profiler.dump_stats('profile.pstats')

    s = io.StringIO()
    sortby = 'cumulative'
    ps = pstats.Stats(profiler, stream=s).sort_stats(sortby)
    ps.print_stats()
    print(s.getvalue())
