import unittest
import pandas as pd
from data_import import RandomTree

class TestRandomTree(unittest.TestCase):
    
    def setUp(self):
        # Erstellen Sie hier ein Objekt der RandomTree Klasse
        # und führen Sie erforderliche Initialisierungslogik aus.
        self.random_tree = RandomTree('energy')

    def test_dataset_columns(self):
        # Stellen Sie sicher, dass die Spalten korrekt ausgelesen werden.
        self.random_tree.get_dataset_columns()
        # Hier sollten weitere Prüfungen der Ausgabe stattfinden

    def test_get_data_from_column(self):
        # Stellen Sie sicher, dass die Daten aus einer Spalte korrekt ausgelesen werden.
        self.random_tree.get_data_from_column(0)
        # Hier sollten weitere Prüfungen der Ausgabe stattfinden

    def test_generate_stratified_test_sets(self):
        # Prüft, ob die Testsets korrekt generiert werden.
        X_train, X_test, y_train, y_test = self.random_tree.generate_stratified_test_sets(0.2, 0)
        # Überprüfen Sie, ob die Testgröße korrekt ist.
        self.assertEqual(len(X_test), int(0.2 * len(self.random_tree.dataframe)))
        self.assertEqual(len(y_test), int(0.2 * len(self.random_tree.dataframe)))
        # Hier sollten weitere Prüfungen der zurückgegebenen Datensätze stattfinden

if __name__ == '__main__':
    unittest.main()