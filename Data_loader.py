import pandas as pd


class Data_Loader:
    """
    A class for loading and preparing data from CSV files.

    """

    def __init__(self, file_path1, file_path2, file_path3):
        """
        Initialize the Data_Loader with file paths.
        """
        self.file_path1 = file_path1
        self.file_path2 = file_path2
        self.file_path3 = file_path3
        self.obs = None
        self.model = None
        self.response = None

    def set_file_paths(self, file_path1, file_path2, file_path3):
        """
        Update file paths for the data files.
        """
        self.file_path1 = file_path1
        self.file_path2 = file_path2
        self.file_path3 = file_path3

    def _read_file(self, file_path):
        """
        Read a CSV file and handle errors.
        """
        try:
            return pd.read_csv(file_path)
        except Exception as e:
            print(f"Error reading file {file_path}: {e}")
            return None

    def read_data(self):
        """
        Load data from all specified file paths.
        """
        self.obs = self._read_file(self.file_path1)
        self.model = self._read_file(self.file_path2)
        self.response = self._read_file(self.file_path3)
        print("Data successfully loaded.")

    def prepare_data(self):
        """
        Prepare data for analysis.
        """
        if self.obs is None or self.model is None or self.response is None:
            print("Data has not been loaded.")
            return None, None, None

        # Example of data preparation
        x_obs = self.obs.iloc[:, 1:]
        y_model = self.model
        x_res = self.response
        return x_obs, y_model, x_res








