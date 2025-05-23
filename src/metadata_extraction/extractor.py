\
import os
import datetime

def get_file_metadata(file_path: str) -> dict | None:
    """
    Extracts metadata from a file, such as size, creation/modification dates, and file type.

    Args:
        file_path: The absolute path to the file.

    Returns:
        A dictionary containing file metadata, or None if an error occurs (e.g., file not found).
    """
    try:
        stat_info = os.stat(file_path)
        metadata = {
            "file_name": os.path.basename(file_path),
            "file_path": os.path.abspath(file_path),
            "file_size_bytes": stat_info.st_size,
            "creation_time_unix": stat_info.st_ctime, # Platform dependent, may be last metadata change time on Unix
            "modification_time_unix": stat_info.st_mtime,
            "access_time_unix": stat_info.st_atime,
            "file_type": os.path.splitext(file_path)[1].lower() # Store extension in lowercase
        }
        # Convert Unix timestamps to ISO 8601 format strings
        metadata["creation_date"] = datetime.datetime.fromtimestamp(metadata["creation_time_unix"]).isoformat()
        metadata["modification_date"] = datetime.datetime.fromtimestamp(metadata["modification_time_unix"]).isoformat()
        metadata["access_date"] = datetime.datetime.fromtimestamp(metadata["access_time_unix"]).isoformat()
        return metadata
    except FileNotFoundError:
        print(f"Error: File not found at {file_path}")
        return None
    except Exception as e:
        print(f"Error extracting metadata for {file_path}: {e}")
        return None

if __name__ == '__main__':
    # Example usage: Get metadata for this script file
    # script_metadata = get_file_metadata(__file__)
    # if script_metadata:
    #     print(f"Metadata for {__file__}:")
    #     for key, value in script_metadata.items():
    #         print(f"  {key}: {value}")
    # else:
    #     print(f"Could not retrieve metadata for {__file__}.")
    pass
