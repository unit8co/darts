import time
import logging
import unittest
import shutil

# Print something for all tests taking longer than this
DURATION_THRESHOLD = 2.


class DartsBaseTestClass(unittest.TestCase):

    @classmethod
    def setUpClass(cls):
        cls.tic = time.time()
        logging.disable(logging.CRITICAL)

    def tearDown(self):
        duration = time.time() - self.tic
        if duration >= DURATION_THRESHOLD:
            print('Test {} finished after {:.2f} s.'.format(self.id(), duration))
        try:
            shutil.rmtree('.darts')
        except:
            pass
