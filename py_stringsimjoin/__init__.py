
__version__ = '0.3.2'

# determine whether to use available cython implementations                     
__use_cython__ = True 

# import join methods
from py_stringsimjoin.join.cosine_join import cosine_join
from py_stringsimjoin.join.dice_join import dice_join
from py_stringsimjoin.join.edit_distance_join import edit_distance_join
from py_stringsimjoin.join.jaccard_join import jaccard_join
from py_stringsimjoin.join.overlap_join import overlap_join
from py_stringsimjoin.join.overlap_coefficient_join import overlap_coefficient_join

# import disk-based join methods
from py_stringsimjoin.join.disk_edit_distance_join import disk_edit_distance_join

# import filters
from py_stringsimjoin.filter.overlap_filter import OverlapFilter
from py_stringsimjoin.filter.position_filter import PositionFilter
from py_stringsimjoin.filter.prefix_filter import PrefixFilter
from py_stringsimjoin.filter.size_filter import SizeFilter
from py_stringsimjoin.filter.suffix_filter import SuffixFilter

# import matcher methods
from py_stringsimjoin.matcher.apply_matcher import apply_matcher

# import profiling methods
from py_stringsimjoin.profiler.profiler import profile_table_for_join

# import utility methods
from py_stringsimjoin.utils.converter import dataframe_column_to_str, series_to_str

# import helper functions
from py_stringsimjoin.utils.generic_helper import get_install_path

# import methods to load sample datasets
from py_stringsimjoin.datasets.base import load_books_dataset, load_person_dataset

# import disk-based join methods
