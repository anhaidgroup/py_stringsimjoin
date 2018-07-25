Disabling Cython 
================

By default (in Linux and OSX), the join methods are implemented in Cython. In 
case there are any issues when invoking the Cython function, we recommend you to 
disable the ``__use_cython__`` flag and try invoking the function again to use 
the Python version of the function. Specifically, Cython usage can be disabled 
using the following command,

    >>> import py_stringsimjoin as ssj
    >>> ssj.__use_cython__ = False

Note that the above command will disable Cython usage for all the join functions.
