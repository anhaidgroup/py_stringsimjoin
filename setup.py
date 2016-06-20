from setuptools import setup, find_packages


setup(
        name='py_stringsimjoin',
        version='0.1.0',
        description='Python library for performing string similarity joins.',
        long_description="""
    String Similarity Join is an important problem in many settings such as data integration, data cleaning,etc.
    This package aims to implement string similarity join over two tables for most commonly used similarity measures such as Jaccard, Dice, Cosine, Overlap, Edit Distance etc.
    """,
        url='http://github.com/anhaidgroup/py_stringsimjoin',
        author='Paul Suganthan G. C.',
        author_email='paulgc@cs.wisc.edu',
        license='BSD',
        packages=find_packages(),
        install_requires=[
            'pandas >= 0.16.0',
            'numpy >= 1.7.0',
            'six',
            'joblib',
            'PyPrind >= 2.9.3',
            'py_stringmatching'
        ],
        include_package_data=True,
        zip_safe=False
)
