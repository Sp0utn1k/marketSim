import logging
import yaml

def setup_logging(config):
    logging.basicConfig(
        filename=config['logging']['file'],
        level=getattr(logging, config['logging']['level']),
        format="%(asctime)s - %(name)s - %(levelname)s - %(message)s"
    )

def load_config(config_file):
    with open(config_file, 'r') as file:
        config = yaml.safe_load(file)
    return config
