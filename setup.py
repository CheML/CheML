from setuptools import setup

setup(name='cheml',
        version='0.0.1',
        description='Machine learning for computational chemistry',
        url='http://github.com/CheML/CheML',
        author='Michael Eickenberg',
        author_email='michael.eickenberg@example.com',
        license='BSD 3-clause',
        packages=['cheml'],    # understand what this means, sklearn-theano puts setuptools.find_packages()
        install_requires=[  'numpy',
                            'scipy',
                            'scikit-learn'],
        classifiers=['Development Status :: 3 - Alpha',
                     'Intended Audience :: Science/Research',
                     'License :: OSI Approved :: BSD License',
                     'Operating System :: OS Independent',
                     'Topic :: Scientific/Engineering']
)


