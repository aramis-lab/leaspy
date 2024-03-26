import os
import ast
from setuptools import setup


def find_version(*py_file_with_version_paths):
    with open(os.path.join(*py_file_with_version_paths), 'r') as fp:
        for line in fp:
            if line.startswith('__version__'):
                return ast.parse(line).body[0].value.s  # string
    raise RuntimeError("Unable to find version string.")


def readme():
    with open('README.md', 'r') as fp:
        return fp.read()


version = find_version("leaspy", "__init__.py")

with open("requirements.txt", 'r') as f:
    requirements = f.read().splitlines()

with open("requirements_dev.txt", 'r') as f:
    dev_requirements = f.read().splitlines()

with open("docs/requirements.txt", 'r') as f:
    docs_requirements = f.read().splitlines()

EXTRAS_REQUIRE = {
    'dev': dev_requirements,
    'docs': docs_requirements
}

setup(name="leaspy",
      version=version,

      description='Leaspy is a software package for the statistical analysis of longitudinal data.',
      long_description=readme(),
      long_description_content_type='text/markdown',
      license='BSD-3-Clause',

      url='https://gitlab.com/icm-institute/aramislab/leaspy',
      project_urls={
          'Bug Reports': 'https://gitlab.com/icm-institute/aramislab/leaspy/issues',
          'Source': 'https://gitlab.com/icm-institute/aramislab/leaspy',
          'Documentation': 'https://leaspy.readthedocs.io',
      },

      author='Igor Koval, Raphael Couronne, Etienne Maheux, Arnaud Valladier, Benoit Martin, Pierre-Emmanuel Poulet, Samuel Gruffaz, Cecile Di Folco, Juliette Ortholand, Mkrtich Vatinyan, Benoit Sauty De Chalon, Nemo Fournier, Quentin Madura, Stanley Durrleman',  # TODO
      author_email='igor.koval@inria.fr',
      maintainer='Igor Koval',
      maintainer_email='igor.koval@inria.fr',

      python_requires='>=3.9, <3.12',

      keywords='leaspy longitudinal mixed-model',

      packages=['leaspy',

                'leaspy.algo',
                'leaspy.algo.fit',
                'leaspy.algo.personalize',
                'leaspy.algo.simulate',
                'leaspy.algo.others',
                'leaspy.algo.utils',
                'leaspy.algo.utils.samplers',

                'leaspy.datasets',

                'leaspy.io',
                'leaspy.io.data',
                'leaspy.io.settings',
                'leaspy.io.realizations',
                'leaspy.io.outputs',
                'leaspy.io.logs',
                'leaspy.io.logs.visualization',

                'leaspy.models',
                'leaspy.models.utils',
                'leaspy.models.utils.attributes',
                'leaspy.models.utils.initialization',

                'leaspy.utils',
                ],

      install_requires=requirements,
      include_package_data=True,
      data_files=[('requirements', ['requirements.txt', 'requirements_dev.txt', 'docs/requirements.txt'])],

      # tests_require=["unittest"],
      test_suite='tests',

      classifiers=[
          # https://pypi.org/classifiers/
          "Development Status :: 5 - Production/Stable",
          "Intended Audience :: Science/Research",
          "Intended Audience :: Developers",
          "Programming Language :: Python :: 3.9",
          "Programming Language :: Python :: 3.10",
          "Programming Language :: Python :: 3.11",
          "License :: OSI Approved :: BSD License",
          "Operating System :: OS Independent",
      ],
      extras_require=EXTRAS_REQUIRE
    )
