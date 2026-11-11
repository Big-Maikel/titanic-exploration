from pathlib import Path

import pandas as pd


DATA_FILE = Path(__file__).parent / "data" / "train.csv"


def clean_train_data() -> None:
	data = pd.read_csv(DATA_FILE)
	data = data.drop(columns=["Ticket"])
	data.to_csv(DATA_FILE, index=False)

if __name__ == "__main__":
	clean_train_data()
