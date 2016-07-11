============
Installation
============
 
Requirements
------------
    * Python 2.7 or Python 3.3+

Platforms
------------
py_stringsimjoin has been tested on Linux (Ubuntu with  Kernel Version 3.13.0-40-generic), OS X (Darwin with Kernel Version 13.4.0), and Windows 8.1.

Dependencies
------------
    * pandas
    * joblib
    * py_stringmatching
    * pyprind
    * six

.. note::

     The py_stringsimjoin installer will automatically install the above required packages. 

There are two ways to install py_stringsimjoin package: using pip or source distribution.

Installing Using pip
--------------------
The easiest way to install the package is to use pip, which will retrieve py_stringsimjoin from PyPI then install it::

    pip install py_stringsimjoin
    
Installing from Source Distribution
-------------------------------------
Step 1: Download the source code of the py_stringsimjoin package from `here
<https://github.com/anhaidgroup/py_stringsimjoin/releases>`_.

Step 2: Unzip or untar the package and execute the following command from the package root::

    python setup.py install
    
.. note::

    The above command will try to install py_stringsimjoin into the defaul Python directory on your machine. If you do not have installation permission for that directory then you can install the package in your home directory as follows::

        python setup.py install --user

    For more information see the StackOverflow `link
    <http://stackoverflow.com/questions/14179941/how-to-install-python-packages-without-root-privileges>`_.
