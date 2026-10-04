import os

class FileDiscoverer:
    """
    Discovers files in a given directory, with options to skip hidden files and directories.
    """
    def __init__(self, skip_hidden: bool = True):
        """
        Initializes the FileDiscoverer.

        Args:
            skip_hidden: Whether to skip hidden files and directories (those starting with '.').
                         Defaults to True.
        """
        self.skip_hidden = skip_hidden

    def scan_directory(self, directory_path: str):
        """
        Recursively scans a directory and yields absolute file paths.

        Args:
            directory_path: The path to the directory to scan.

        Yields:
            Absolute path to each discovered file.
        """
        for root, dirs, files in os.walk(os.path.abspath(directory_path)):
            if self.skip_hidden:
                # Filter out hidden directories from os.walk's list of dirs to traverse
                dirs[:] = [d for d in dirs if not d.startswith('.')]
            
            for file_name in files:
                if self.skip_hidden and file_name.startswith('.'):
                    continue
                yield os.path.join(root, file_name)
