from tabulate import tabulate
import pandas as pd
import numpy as np

class Verbose:
    """
    Arbitrary printing class. 
    This class is used to print messages.
    Only if verbose is True, the message will be printed.
    """
    def __init__(self, verbose: bool = True):
        self.verbose = verbose
    
    def print(self, message: str):
        if self.verbose:
            print(message)

    def print_table_raw_data(self, data_df):
        headers = ["Data Type"]+[key for key in data_df.keys()]
        table_data = [
            ["Raw Data"] + [f"{data_df[key].shape if data_df[key] is not None else 'N/A'}" for key in data_df.keys()],
            ["#NaN"] + [f"{data_df[key].isna().sum().sum() if data_df[key] is not None else 'N/A'}" for key in data_df.keys()],
            ["#0's"] + [f"{(data_df[key].select_dtypes(include=[np.number]) == 0).sum().sum() if data_df[key] is not None else 'N/A'}" for key in data_df.keys()]
        ]
        self.print(f"\n📊 Raw Data Summary:")
        self.print(tabulate(table_data, headers=headers, tablefmt="grid"))
        
        

    def print_table_split_data(self, data_df):
            table_data = [
            ["Features (X)"] + [f"{data_df[0][key].shape if data_df[0][key] is not None else 'N/A'}" for key in data_df[0].keys()],
            ["Target (y)"] + [f"{data_df[2][key].shape if data_df[2][key] is not None else 'N/A'}" for key in data_df[2].keys()],
            ["Demographics"] + [f"{data_df[1][key].shape if data_df[1][key] is not None else 'N/A'}" for key in data_df[1].keys()],
            ["HIV+ ratio"] + [f"{data_df[2][key].mean():.2f}" if data_df[2][key] is not None else 'N/A' for key in data_df[2].keys()],
            ["#NaN"] + [f"{data_df[0][key].isna().sum().sum() if data_df[0][key] is not None else 'N/A'}" for key in data_df[0].keys()],
            ["#0's"] + [f"{(data_df[0][key].select_dtypes(include=[np.number]) == 0).sum().sum() if data_df[0][key] is not None else 'N/A'}" for key in data_df[0].keys()],
            ["#<0"] + [f"{(data_df[0][key].select_dtypes(include=[np.number]) < 0).sum().sum() if data_df[0][key] is not None else 'N/A'}" for key in data_df[0].keys()],
            ["#columns with non-numeric values"] + [f"{len(data_df[0][key].select_dtypes(exclude=[np.number]).columns.tolist()) if data_df[0][key] is not None else 'N/A'}" for key in data_df[0].keys()]
        ]
        
            headers = ["Data Type"] + [key for key in data_df[0].keys()]
            self.print(f"\n📊 Raw Data Summary after splitting and dropping unlabelled:")
            self.print(tabulate(table_data, headers=headers, tablefmt="grid"))

    def print_stats(self, X: pd.DataFrame) -> None:
        """Print statistical summary of the DataFrame"""
        if X.empty or len(X.columns) == 0:
            self.print("\n" + "+"*20 + " Summary of the last 5 columns " + "+"*20)
            self.print("No columns remaining after quality control")
            return
            
        columns = X.columns.tolist()[-5:]
        table = X[columns].describe().T
        header = ['count', 'mean', 'std', 'min', '25%', '50%', '75%', 'max']
        self.print("\n" + "+"*20 + " Summary of the last 5 columns " + "+"*20)
        self.print(tabulate(table, headers=header, tablefmt='grid', floatfmt='.3f'))
    

        # second table with one row, showing the # Nan, # 0, #>1e4, #<1e-4, Shape
        if X.empty or len(X.columns) == 0:
            summary_table = pd.DataFrame({
                'shape': [X.shape],
                '#null': 0,
                '#0': 0,
                '# >1e5': 0,
                '# <1e-5': 0,
                '#unique \n mean': 0,
                '#unique \n std': 0,
                '#unique \n mode': 0,
            })
        else:
            summary_table = pd.DataFrame({
                'shape': [X.shape],
                '#null': X.isnull().sum().sum(),
                '#0': X.isin([0]).sum().sum(),
                '# >1e5': (X > 1e5).sum().sum(),
                '# <1e-5': (np.abs(X) < 1e-4).sum().sum(),
                '#unique \n mean': X.nunique(axis=0).mean(),
                '#unique \n std': X.nunique(axis=0).std(),
                '#unique \n mode': X.mode(axis=0).iloc[0].nunique(),
            })
        self.print("\n" + "+"*20 + " Final summary " + "+"*20)
        self.print(tabulate(summary_table, headers='keys', tablefmt='grid', floatfmt='.3f'))

    def set_verbose(self, verbose: bool):
        self.verbose = verbose
    
    def get_verbose(self):
        return self.verbose