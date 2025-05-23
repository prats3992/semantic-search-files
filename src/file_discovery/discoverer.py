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
        # print(f"Scanning directory: {directory_path}") # Optional: for verbose logging
        for root, dirs, files in os.walk(os.path.abspath(directory_path)):
            if self.skip_hidden:
                # Filter out hidden directories from os.walk's list of dirs to traverse
                dirs[:] = [d for d in dirs if not d.startswith('.')]
            
            for file_name in files:
                if self.skip_hidden and file_name.startswith('.'):
                    # print(f"Skipping hidden file: {os.path.join(root, file_name)}") # Optional
                    continue
                file_path = os.path.join(root, file_name)
                # print(f"Found file: {file_path}") # Optional
                yield file_path

            # Optional: for verbose logging of explicitly skipped hidden directories
            # if self.skip_hidden:
            #     original_dirs = set(os.listdir(root) if os.path.isdir(root) else [])
            #     current_dirs_to_walk = set(dirs)
            #     for d_name in original_dirs - current_dirs_to_walk:
            #         if d_name.startswith('.') and os.path.isdir(os.path.join(root, d_name)):
            #             print(f"Skipping hidden directory traversal: {os.path.join(root, d_name)}")

if __name__ == '__main__':
    # Example usage:
    discoverer_all = FileDiscoverer(skip_hidden=False)
    discoverer_no_hidden = FileDiscoverer(skip_hidden=True)
    
    # Create some dummy files and directories for testing
    # test_scan_dir = "./temp_scan_test_dir"
    # os.makedirs(os.path.join(test_scan_dir, ".hidden_subdir"), exist_ok=True)
    # open(os.path.join(test_scan_dir, "visible_file.txt"), 'a').close()
    # open(os.path.join(test_scan_dir, ".hidden_file.txt"), 'a').close()
    # open(os.path.join(test_scan_dir, ".hidden_subdir", "another_visible.txt"), 'a').close()
    # open(os.path.join(test_scan_dir, ".hidden_subdir", ".deep_hidden.txt"), 'a').close()

    # print(f"Scanning '{test_scan_dir}' (including hidden):")
    # for f_path in discoverer_all.scan_directory(test_scan_dir):
    #     print(f"  - {f_path}")

    # print(f"\\nScanning '{test_scan_dir}' (skipping hidden):")
    # for f_path in discoverer_no_hidden.scan_directory(test_scan_dir):
    #     print(f"  - {f_path}")

    # # Clean up dummy files and directories
    # import shutil
    # if os.path.exists(test_scan_dir):
    #     shutil.rmtree(test_scan_dir)
    pass
