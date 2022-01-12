from distutils.core import setup

with open("README.md", "r", encoding="utf-8") as fh:
    long_description = fh.read()

setup(name="mlalgo",
      version="0.3",
      description="Machine Learning ToolKit.",
      long_description=long_description,
      long_description_content_type="text/markdown",
      author="Zeyad Khaled",
      url="https://github.com/zeyad-kay/mlalgo",
      project_urls={
          "Bug Tracker": "https://github.com/zeyad-kay/mlalgo/issues",
      },
      packages=["mlalgo"],
      requires=["numpy"],
      classifiers=[
          "Programming Language :: Python :: 3",
          "License :: OSI Approved :: MIT License",
          "Operating System :: OS Independent",
      ],
      )
