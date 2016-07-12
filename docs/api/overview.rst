Overview
========

Given two tables A and B, this package provides commands to perform string similarity joins between two columns of these tables, such as A.name and B.name, or A.city and B.city. An example of such joins is to return all pairs (x,y) of tuples from the Cartesian product of Tables A and B such that 

* x is a tuple in Table A and y is a tuple in Table B.
* Jaccard(3gram(x.name), 3gram(y.name)) > 0.7. That is, first tokenize the value of the attribute "name" of x into a set P of 3grams, and tokenize the value of the attribute "name" of y into a set Q of 3grams. Then compute the Jaccard score between P and Q. This score must exceed 0.7. This is often called the "join condition". 

Such joins are challenging because a naive implementation would consider all tuple pairs in the Cartesian product of Tables A and B, an often enormous number (for example, 10 billion pairs if each table has 100K tuples). The package provides efficient implementations of such joins, by using methods called "filtering" to quickly eliminate the pairs that obviously cannot satisfy the join condition. 

To understand tokenizing and string similarity scores (such as Jaccard, edit distance, etc.), see the Web site of the package py_stringmatching (in particular, read the following book chapter on string matching). That package provides efficient implementations of a set of tokenizers and string similarity measures. It focuses on the case of tokenizing two strings and then applying a similarity measure to the outputs of the tokenizers to compute a similarity score between those two strings. This package builds on top of py_stringmatching, so it is important to understand the tokenizers and string similarity measures of the py_stringmatching package. 

To read more about string similarity joins, see [[which paper to mention here? Perhaps "string similarity joins: an experimental evaluation?". Is there a better paper? Maybe mention two papers]]

We now explain the most important notions an user is likely to encounter while using this package. To use the package, the user typically loads into Python two tables A and B (as described above). These two tables will often be referred to in the commands of this package as ltable (for "left table") and rtable (for "right table"), respectively. The notion "tuple pair" refers to a pair (x,y) where x is a tuple in ltable and y is a tuple in rtable. 

To execute a string similarity join, the user calls a command such as jaccard_join, cosine_join, etc. The command's arguments include

* The two tables: ltable and rtable, the key attributes of these tables, and the target attributes (on which the join will be performed, such as A.name and B.name).
* The join condition.
* The output will be a table of tuple pairs surviving the join. The user may want to specify the desired attributes of the output table, using arguments such as l_out_attrs, r_out_attrs, l_out_prefix, etc.
* A flag to indicate on how many cores should this command be run. 

Internally, a command such as jaccard_join will first create a filter object, using an appropriate filtering techniques (many such techniques exist, see the book chapter on string matching). Next, it uses this filter object to quickly drop many pairs that obviously do not satisfy the join condition. The set of remaining tuple pairs is referred to as a "candidate set". Finally, it applies a matcher to the pairs in this set. The matcher simply checks and retains only those pairs that satisfy the join condition. 

The implemented commands can be organized into the following groups: profilers, joins, filters, matchers, and utilities. We now briefly describe these. 

Profilers
---------
After loading the two tables A and B into Python, the user may want to run a profiler command, which will examine the two tables, detect possible problems for the subsequent string similarity joins, warn the user of these potential problems, and suggest possible solutions. 

Currently only one profiler has been implemented, profile_table_for_join. This command examines the tables for unique and missing attribute values, and discuss possible problems stemming from these for the subsequent joins. Based on the report of this command, the user may want to take certain actions before actually applying join commands. Using the profiler is not required, of course. 

Joins
-----
After loading and optionally profiling and fixing the tables, most likely the user will just call a join command to do the join. We have implemented the following join commands:  

* cosine_join
* dice_join
* edit_distance_join
* jaccard_join
* overlap_coefficient_join
* overlap_join


Filters
-------

Version 0.1.0 implements the following class hierarchy for filters:

Filter                                                               
  * OverlapFilter
  * SizeFilter
  * PrefixFilter
  * PositionFilter
  * SuffixFilter

Matchers
-------

Version 0.1.0 implements the following method for applying a matcher:

apply_matcher
