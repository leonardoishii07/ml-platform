import yaml
from src.ingest.loader_factory import get_loader

# Load config
with open("configs/dataset1.yaml") as f:
    config = yaml.safe_load(f)

# Load data
loader = get_loader(config)
df = loader.load()

print(df.head())
