from dtu import setup

# see 'module available' on server for newest python version
setup(
    f"https://github.com/{user}/{project}.git",
    python="3.9.6",
    packages=["torch", "torchvision", "matplotlib"]
)
