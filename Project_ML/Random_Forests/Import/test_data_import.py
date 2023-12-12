import unittest
import pandas as pd
from data_import import DatasetImport
import tempfile
import os

class TestDatasetImport(unittest.TestCase):
    def setUp(self):
        self.temp_dir = tempfile.TemporaryDirectory()
        self.temp_files = []
        for i in range(4):
            file_path = os.path.join(self.temp_dir.name, f"temp_data_{i}.csv")
            with open(file_path, 'w') as f:
                f.write(f"A,B,C,D\n1,2,3,{i+1}\n5,6,7,{i+1}\n9,10,11,{i+1}\n")
            self.temp_files.append(file_path)
            if i == 0:
                # Erstellt die DatasetImport-Instanz basierend auf dem Dateinamen ohne die ".csv" Erweiterung.
                self.data_import = DatasetImport(file_path[:-4])

    def tearDown(self):
        self.temp_dir.cleanup()  # Entfernt das Verzeichnis und alle darin enthaltenen Dateien.

    def test_import_data(self):
        self.assertEqual(self.data_import.dataframe.shape, (3, 4))

    def test_get_recommended_subset(self):
        subset = self.data_import.get_recommended_subset(2)
        self.assertEqual(subset, 2)

# ... (Weitere Testfälle)
if __name__ == '__main__':
    unittest.main()
