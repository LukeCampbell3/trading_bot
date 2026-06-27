import threading
import os
from glob import glob
import tensorflow as tf
from simulation import Simulation

# Safe GPU memory handling
try:
    gpus = tf.config.list_physical_devices('GPU')
    if gpus:
        tf.config.experimental.set_memory_growth(gpus[0], True)
except Exception as e:
    print(f"TensorFlow GPU config error: {e}")

class MultiFileRunner:
    def __init__(self, feather_dir, model_path, multi_stock=False):
        self.feather_dir = feather_dir
        self.model_path = model_path
        self.multi_stock = multi_stock
        self.file_groups = self._group_files_by_day()

    def _group_files_by_day(self):
        feather_files = sorted(glob(os.path.join(self.feather_dir, "*.feather")))
        return {os.path.basename(f): os.path.dirname(f) for f in feather_files}

    def run_simulation(self, symbol, folder_path):
        print(f"Running: {symbol}")
        sim = Simulation(feather_folder=folder_path, model_path=self.model_path, symbol=symbol, use_gpu=True)
        sim.run()

    def run(self):
        if not self.multi_stock:
            symbol, path = next(iter(self.file_groups.items()))
            self.run_simulation(symbol, path)
        else:
            threads = []
            for symbol, path in self.file_groups.items():
                t = threading.Thread(target=self.run_simulation, args=(symbol, path))
                threads.append(t)
                t.start()
            for t in threads:
                t.join()

if __name__ == "__main__":
    runner = MultiFileRunner(
        feather_dir="HFT\\data_by_days",
        model_path="HFT\\archive\\transformer_model.keras",
        multi_stock=False
    )
    runner.run()