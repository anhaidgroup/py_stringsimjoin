import subprocess
import sys
import os
import re
import contextlib

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
from distutils import ccompiler, msvccompiler
from distutils.sysconfig import get_python_inc


## fix compiler and build options
COMPILE_OPTIONS = {
        'msvc': ['/0x', '/EHsc'],
        'mingw32': ['-O3', '-ffast-math', '-march=native'],
        'other': ['-O3', '-ffast-math', '-march=native']
}

LINK_OPTIONS = {
        'msvc': [],
        'mingw32': [],
        'other':[]
}

class build_ext_options:
    def build_options(self):
        for e in self.extensions:
            e.extra_compile_args += COMPILE_OPTIONS.get(
                self.compiler.compiler_type, COMPILE_OPTIONS['other'])
        for e in self.extensions:
            e.extra_link_args += LINK_OPTIONS.get(
                    self.compiler.compiler_type, LINK_OPTIONS['other'])

class build_ext(_build_ext, build_ext_options):
    def build_extensions(self):
        build_ext_options.build_options(self)
        _build_ext.build_extensions(self)


def generate_cython():
    cwd = os.path.abspath(os.path.dirname(__file__))
    print("Cythonizing sources")
    p = subprocess.call([sys.executable, os.path.join(cwd,
                             'build_tools',
                             'cythonize.py'),
            'py_stringsimjoin'],
        cwd=cwd)
    if p != 0:
        raise RuntimeError("Running cythonize failed!")

MODULES = {
        "py_stringsimjoin.index.inverted_index_cy": {'sources':["py_stringsimjoin/index/inverted_index_cy.pyx"],
                                                        'comargs':[]            
                                                        },  
        "py_stringsimjoin.index.position_index_cy": {'sources':["py_stringsimjoin/index/position_index_cy.pyx"],
                                                        'comargs':[]            
                                                        },  
        "py_stringsimjoin.similarity_measure.edit_distance": {'sources':["py_stringsimjoin/similarity_measure/edit_distance.pyx"],
                                                        'comargs':[]
                                                        },

        "py_stringsimjoin.similarity_measure.cosine": {'sources':["py_stringsimjoin/similarity_measure/cosine.pyx"],
                                                        'comargs':[]            
                                                        }, 

        "py_stringsimjoin.similarity_measure.dice": {'sources':["py_stringsimjoin/similarity_measure/dice.pyx"],
                                                        'comargs':[]            
                                                        }, 

        "py_stringsimjoin.similarity_measure.jaccard": {'sources':["py_stringsimjoin/similarity_measure/jaccard.pyx"],
                                                        'comargs':[]            
                                                        }, 

        "py_stringsimjoin.join.edit_distance_join_cy": {'sources':["py_stringsimjoin/join/edit_distance_join_cy.pyx",
                                                                   ],
                                                        'comargs':["-I./py_stringsimjoin/index/"]
                                                        },
        "py_stringsimjoin.join.disk_edit_distance_join_cy": {'sources': ["py_stringsimjoin/join/disk_edit_distance_join_cy.pyx",
                    ],
        'comargs': ["-I./py_stringsimjoin/index/"]
        },

        "py_stringsimjoin.join.overlap_coefficient_join_cy": {'sources':["py_stringsimjoin/join/overlap_coefficient_join_cy.pyx",
                                                                    ],
                                                        'comargs':["-I./py_stringsimjoin/index/"]
                                                        },

        "py_stringsimjoin.join.overlap_join_cy": {'sources':["py_stringsimjoin/join/overlap_join_cy.pyx",
                                                             ],
                                                        'comargs':["-I./py_stringsimjoin/index/"]
                                                        },

        "py_stringsimjoin.join.cosine_join_cy": {'sources':["py_stringsimjoin/join/cosine_join_cy.pyx"],
                                                        'comargs':["-I./py_stringsimjoin/index/"]
                                                        },  

        "py_stringsimjoin.join.dice_join_cy": {'sources':["py_stringsimjoin/join/dice_join_cy.pyx"],
                                                        'comargs':["-I./py_stringsimjoin/index/"]
                                                        },  

        "py_stringsimjoin.join.jaccard_join_cy": {'sources':["py_stringsimjoin/join/jaccard_join_cy.pyx"],
                                                        'comargs':["-I./py_stringsimjoin/index/"]
                                                        },       

        "py_stringsimjoin.join.set_sim_join_cy": {'sources':["py_stringsimjoin/join/set_sim_join_cy.pyx",
                                                             ],
                                                        'comargs':["-I./py_stringsimjoin/index/"]
                                                        },   

        "py_stringsimjoin.utils.cython_utils": {'sources': ["py_stringsimjoin/utils/cython_utils.pyx", 
                                                            ],
                                               'comargs': ["-I./py_stringsimjoin/index/"]
                                               }
}


def is_source_release(path):
    return os.path.exists(os.path.join(path, 'PKG-INFO'))

def purge(dir, pattern):
    for f in os.listdir(dir):
        if re.search(pattern, f):
            file_path=os.path.join(dir,f)
            os.remove(file_path)

def clean(path):
    for name in list(MODULES.keys()):
        dir_list = name.split('.')[0:-1]
        dir_name = '/'.join(dir_list)
        purge(dir_name, r".*\.so$")

        name = name.replace('.','/')
        for ext in ['.cpp', '.c']:
            file_path = os.path.join(path, name + ext)
            if os.path.exists(file_path):
                os.unlink(file_path)




@contextlib.contextmanager
def chdir(new_dir):
    old_dir = os.getcwd()
    try:
        os.chdir(new_dir)
        sys.path.insert(0, new_dir)
        yield
    finally:
        del sys.path[0]
        os.chdir(old_dir)


def setup_package():
    root = os.path.abspath(os.path.dirname(__file__))

    if len(sys.argv) > 1 and sys.argv[1] == 'clean':
        return clean(root)
    if len(sys.argv) > 1 and sys.argv[1] == 'touch':
        return touch(root)


    with chdir(root):
        include_dirs = [get_python_inc(plat_specific=True)]

        if (ccompiler.new_compiler().compiler_type == 'msvc'
                and msvccompiler.get_build_version == 9):
            include_dirs.append(os.path.join(root, 'include', 'msvc9'))

        if not is_source_release(root):
            generate_cython()

        extensions = []
        for name in list(MODULES.keys()):
            curr_mod = MODULES[name]
            e = setuptools.Extension(name, sources=curr_mod['sources'],
                    extra_compile_args=curr_mod['comargs'], language='c++')
            extensions.append(e)



        packages = setuptools.find_packages()
        with open('README.rst') as f:
            LONG_DESCRIPTION = f.read()

        cmdclass = {"build_ext": build_ext}
        setuptools.setup(
            name='py_stringsimjoin',
            version='0.3.1',
            description='Python library for performing string similarity joins.',
            long_description=LONG_DESCRIPTION,
            url='https://sites.google.com/site/anhaidgroup/projects/magellan/py_stringsimjoin',
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
                'Programming Language :: Python :: 3.5',
                'Programming Language :: Python :: 3.6',
                'Programming Language :: Python :: 3.7',                        
                'Topic :: Scientific/Engineering',
                'Topic :: Utilities',
                'Topic :: Software Development :: Libraries',
            ],
            packages=packages,
            ext_modules=extensions,
            cmdclass=cmdclass,
            install_requires=[
                'joblib',
                'pandas >= 0.16.0',
                'PyPrind >= 2.9.3',
                'py_stringmatching >= 0.2.1',
                'six'
            ],
            include_package_data=True,
            zip_safe=False,
        )
if __name__ == '__main__':
    setup_package()

