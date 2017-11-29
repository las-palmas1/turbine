from distutils.core import setup

setup(
    name='turbine',
    version='0.0.1',
    package_dir={'turbine': 'core', 'turbine.average_streamline': 'core/average_streamline',
                 'turbine.profiling': 'core/profiling', 'turbine.cooling': 'core/cooling'},
    packages=['turbine', 'turbine.average_streamline', 'turbine.profiling', 'turbine.cooling'],
    url='',
    license='',
    author='Alexander Zhigalkin',
    author_email='aszhigalkin94@gmail.com',
    description='Library for computing parameters of gas turbine along average streamline.'
)

