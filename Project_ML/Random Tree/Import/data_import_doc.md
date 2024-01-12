Die `RandomTree` Klasse ermöglicht die Bereinigung und Vorverarbeitung von Datensätzen für maschinelles Lernen. Sie bietet Methoden, um Daten zu importieren, zu säubern, zu inspizieren und in Trainings- und Testsets aufzuteilen. 

#### Methoden:
- `__init__(self, data: str)`: Konstruktor der Klasse, der einen Dateipfad als Zeichenkette erwartet. Importiert und säubert Daten und initialisiert die `subset_value` Variable, welche die geschätzte Anzahl der Features für den Random Forest angibt.

- `import_data(self, data: str)`: Importiert Daten aus einer CSV-Datei und säubert sie, indem sie alle Zeilen mit fehlenden Werten eliminiert.
- `drop_column(self, column_index: int)`: Entfernt eine Spalte basierend auf ihrem Index.
- `get_dataset_columns(self)`: Gibt die Spaltennamen des DataFrames aus.
- `get_data_from_column(self, column_index: int)`: Gibt die Daten einer Spalte aus.
- `generate_stratified_test_sets(self, test_size: float, target: int)`: Teilt den DataFrame in stratifizierte Trainings- und Testsets auf, basierend auf der angegebenen Zielvariable und dem Testgrößenparameter.

#### Beispiel:
```python
# Erstellt ein RandomTree-Objekt mit dem CSV-Datensatz 'energy.csv'.
random_tree = RandomTree('energy')
# Generiert Trainings- und Testsets mit dem ersten Feature als Zielvariable.
X_train, X_test, y_train, y_test = random_tree.generate_stratified_test_sets(0.2, 0)