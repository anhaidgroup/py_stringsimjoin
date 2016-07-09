"""
Base IO code for all datasets
"""

from os.path import dirname, join

import pandas as pd


def load_books_dataset():
    """Load and return the books dataset.
    """
    base_dir = join(dirname(__file__), 'data')
    table_A = pd.read_csv(join(base_dir, 'books_table_A.csv.gz'),
                          compression='gzip', header=0)
    table_B = pd.read_csv(join(base_dir, 'books_table_B.csv.gz'),
                          compression='gzip', header=0)
    return table_A, table_B


def load_person_dataset():                                                       
    """Load and return the person dataset.                                       
    """                                                                         
    base_dir = join(dirname(__file__), 'data')                                  
    table_A = pd.read_csv(join(base_dir, 'person_table_A.csv'))                         
    table_B = pd.read_csv(join(base_dir, 'person_table_B.csv'))                         
    return table_A, table_B   
