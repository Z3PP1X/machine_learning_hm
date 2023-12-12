import unittest
import pandas as pd
from data_import import DatasetImport
import tempfile
import os

class TestDatasetImport(unittest.TestCase):
    def setUp(self):
        self.temp_files = []
        for i in range(4):
            temp, path = tempfile.mkstemp(suffix='.csv')
            os.write(temp, f"A,B,C,D\n1,2,3,{i+1}\n5,6,7,{i+1}\n9,10,11,{i+1}\n".encode())
            os.close(temp)
            self.temp_files.append(path)
            if i == 0:
                self.data_import = DatasetImport(path[:-4])

    def tearDown(self):
        for f in self.temp_files:
            os.remove(f)

    def test_import_data(self):
        self.assertEqual(self.data_import.dataframe.shape, (3, 4))

    def test_get_recommended_subset(self):
        subset = self.data_import.get_recommended_subset(2)
        self.assertEqual(subset, 2)

# ... (Weitere Testfälle)
if __name__ == '__main__':
    unittest.main()
