import time
import logging
import unittest
import shutil


class DartsBaseTestClass(unittest.TestCase):
    @classmethod
    def setUpClass(cls):
        cls.tic = time.time()
        logging.disable(logging.CRITICAL)

    def tearDown(self):
        print('Test {} finished after {:.2f} s.'.format(self.id(), time.time() - self.tic))
        try:
            shutil.rmtree('.darts')
        except:
            pass
