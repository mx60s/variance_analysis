# variance_analysis/setup.py

from setuptools import setup, find_packages

def main():
    setup(
        name="variance_analysis",
        version="0.1.0",
        packages=find_packages(),
        install_requires=[
            "numpy",
            "Pillow",
            "matplotlib",
            "seaborn",
            "scipy",
            "tqdm",
            "scikit-learn",
        ],
    )

if __name__ == "__main__":
    main()

