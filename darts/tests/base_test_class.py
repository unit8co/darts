import logging
import shutil
import time
import unittest

# Print something for all tests taking longer than this
DURATION_THRESHOLD = 2.0
tfm_kwargs = {"pl_trainer_kwargs": {"accelerator": "cpu"}}


class DartsBaseTestClass(unittest.TestCase):
    @classmethod
    def setUpClass(cls):
        cls.tic = time.time()
        logging.disable(logging.CRITICAL)

    def tearDown(self):
        duration = time.time() - self.tic
        if duration >= DURATION_THRESHOLD:
            print(f"Test {self.id()} finished after {duration:.2f} s.")
        try:
            shutil.rmtree(".darts")
        except FileNotFoundError:
            pass
