
import subprocess                                                               
import sys                                                                      
import os 

# check if pip is installed. If not, raise an ImportError
PIP_INSTALLED = True

try:
    import pip
except ImportError:
    PIP_INSTALLED = False

if not PIP_INSTALLED:
    raise ImportError('pip is not installed.')

def install_and_import(package):
    import importlib
    try:
        importlib.import_module(package)
    except ImportError:
        pip.main(['install', package])
    finally:
        globals()[package] = importlib.import_module(package)

# check if setuptools is installed. If not, install setuptools
# automatically using pip.
install_and_import('setuptools')

from setuptools.command.build_ext import build_ext as _build_ext                
                                                                                
class build_ext(_build_ext):                                                    
    def build_extensions(self):                                                 
        _build_ext.build_extensions(self)                                       
                                                                                
def generate_cython():                                                          
    cwd = os.path.abspath(os.path.dirname(__file__))                            
    print("Cythonizing sources")                                                
    p = subprocess.call([sys.executable, os.path.join(cwd,                      
                                                      'build_tools',            
                                                      'cythonize.py'),          
                         'cythontest'],                                         
                        cwd=cwd)                                                
    if p != 0:                                                                  
        raise RuntimeError("Running cythonize failed!")                         
                                                                                
                                                                                
cmdclass = {"build_ext": build_ext}   

if __name__ == "__main__":

    no_frills = (len(sys.argv) >= 2 and ('--help' in sys.argv[1:] or            
                                         sys.argv[1] in ('--help-commands',     
                                                         'egg_info', '--version',
                                                         'clean')))             
                                                                                
    cwd = os.path.abspath(os.path.dirname(__file__))                            
    if not os.path.exists(os.path.join(cwd, 'PKG-INFO')) and not no_frills:     
        # Generate Cython sources, unless building from source release          
        generate_cython()                                                       

    edit_dist_ext = setuptools.Extension("py_stringsimjoin.similarity_measure.edit_distance",
        sources=["py_stringsimjoin/similarity_measure/edit_distance.pyx"], 
        language="c++",  
        extra_compile_args = ["-O3", "-ffast-math", "-march=native"])   

    edit_dist_join_ext = setuptools.Extension("py_stringsimjoin.join.edit_distance_join_cy",
        sources=["py_stringsimjoin/join/edit_distance_join_cy.pyx",                   
                 "py_stringsimjoin/index/inverted_index_cy.cpp"],               
        language="c++",                                                         
        extra_compile_args = ["-I./py_stringsimjoin/index/", "-O3",             
                              "-ffast-math", "-march=native", "-fopenmp"],      
        extra_link_args=["-fopenmp"])     

    cython_utils_ext = setuptools.Extension("py_stringsimjoin.utils.cython_utils",
        sources=["py_stringsimjoin/utils/cython_utils.pyx"],                    
        language="c++",                                                         
        extra_compile_args = ["-O3", "-ffast-math", "-march=native"])      

    # specify extensions that need to be compiled                               
    extensions = [edit_dist_ext, 
                  edit_dist_join_ext,
                  cython_utils_ext]

    # find packages to be included.
    packages = setuptools.find_packages()

    with open('README.rst') as f:
        LONG_DESCRIPTION = f.read()

    setuptools.setup(
        name='py_stringsimjoin',
        version='0.1.0',
        description='Python library for performing string similarity joins.',
        long_description=LONG_DESCRIPTION,
        url='https://sites.google.com/site/anhaidgroup/projects/py_stringsimjoin',
        author='UW Magellan Team',
        author_email='uwmagellan@gmail.com',
        license='BSD',
        classifiers=[
            'Development Status :: 4 - Beta',
            'Environment :: Console',
            'Intended Audience :: Developers',
            'Intended Audience :: Science/Research',
            'Intended Audience :: Education',
            'License :: OSI Approved :: BSD License',
            'Operating System :: POSIX',
            'Operating System :: Unix',
            'Operating System :: MacOS',
            'Operating System :: Microsoft :: Windows',
            'Programming Language :: Python',
            'Programming Language :: Python :: 2',
            'Programming Language :: Python :: 3',
            'Programming Language :: Python :: 2.7',
            'Programming Language :: Python :: 3.3',
            'Programming Language :: Python :: 3.4',
            'Programming Language :: Python :: 3.5',
            'Topic :: Scientific/Engineering',
            'Topic :: Utilities',
            'Topic :: Software Development :: Libraries',
        ],
        packages=packages,
        install_requires=[
            'joblib', 
            'pandas >= 0.16.0',
            'PyPrind == 2.9.8',
            'py_stringmatching >= 0.2.1',
            'six'                                                              
        ],
        include_package_data=True,
        zip_safe=False,
        ext_modules=extensions,                                                 
        cmdclass=cmdclass,                                                      
    )
