import pathlib

def repo_dir():
	return str(pathlib.Path(__file__).parent.parent.resolve())