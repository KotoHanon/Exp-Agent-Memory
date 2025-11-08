"""
File operation tools for experiment agents.

Provides tools for reading, writing, and managing files and directories.
Compatible with openai-agents SDK.
"""

import os
import shutil
from pathlib import Path
from typing import Dict, Any, List, Optional

from agents import function_tool


@function_tool
def read_file(file_path: str, encoding: str = "utf-8") -> Dict[str, Any]:
    """
    Read content from a file.

    Args:
        file_path: Path to the file to read
        encoding: File encoding (default: utf-8)

    Returns:
        Dictionary with success status, content, and metadata
    """
    try:
        file_path = os.path.expanduser(file_path)
        with open(file_path, "r", encoding=encoding) as f:
            content = f.read()

        file_size = os.path.getsize(file_path)
        line_count = content.count("\n") + 1

        return {
            "success": True,
            "content": content,
            "file_path": file_path,
            "size_bytes": file_size,
            "line_count": line_count,
        }
    except FileNotFoundError:
        return {
            "success": False,
            "error": f"File not found: {file_path}",
        }
    except Exception as e:
        return {
            "success": False,
            "error": f"Error reading file: {str(e)}",
        }


@function_tool
def write_file(
    file_path: str, content: str, encoding: str = "utf-8", create_dirs: bool = True
) -> Dict[str, Any]:
    """
    Write content to a file.

    Args:
        file_path: Path to the file to write
        content: Content to write
        encoding: File encoding (default: utf-8)
        create_dirs: Create parent directories if they don't exist

    Returns:
        Dictionary with success status and message
    """
    try:
        file_path = os.path.expanduser(file_path)

        if create_dirs:
            os.makedirs(os.path.dirname(file_path), exist_ok=True)

        with open(file_path, "w", encoding=encoding) as f:
            f.write(content)

        file_size = os.path.getsize(file_path)

        return {
            "success": True,
            "message": f"Successfully wrote {file_size} bytes to {file_path}",
            "file_path": file_path,
            "size_bytes": file_size,
        }
    except Exception as e:
        return {
            "success": False,
            "error": f"Error writing file: {str(e)}",
        }


@function_tool
def list_directory(
    directory_path: str, pattern: Optional[str] = None, recursive: bool = False
) -> Dict[str, Any]:
    """
    List files and directories in a directory.

    Args:
        directory_path: Path to the directory
        pattern: Optional glob pattern to filter files (e.g., "*.py")
        recursive: List files recursively

    Returns:
        Dictionary with success status and list of files/directories
    """
    try:
        directory_path = os.path.expanduser(directory_path)
        path = Path(directory_path)

        if not path.exists():
            return {
                "success": False,
                "error": f"Directory not found: {directory_path}",
            }

        if not path.is_dir():
            return {
                "success": False,
                "error": f"Path is not a directory: {directory_path}",
            }

        if recursive and pattern:
            files = [str(p) for p in path.rglob(pattern)]
        elif recursive:
            files = [str(p) for p in path.rglob("*")]
        elif pattern:
            files = [str(p) for p in path.glob(pattern)]
        else:
            files = [str(p) for p in path.iterdir()]

        # Separate files and directories
        file_list = []
        dir_list = []
        for item in files:
            item_path = Path(item)
            if item_path.is_file():
                file_list.append(
                    {
                        "path": str(item),
                        "name": item_path.name,
                        "size": item_path.stat().st_size,
                    }
                )
            elif item_path.is_dir():
                dir_list.append(
                    {
                        "path": str(item),
                        "name": item_path.name,
                    }
                )

        return {
            "success": True,
            "directory": directory_path,
            "files": file_list,
            "directories": dir_list,
            "total_files": len(file_list),
            "total_directories": len(dir_list),
        }
    except Exception as e:
        return {
            "success": False,
            "error": f"Error listing directory: {str(e)}",
        }


@function_tool
def create_directory(directory_path: str) -> Dict[str, Any]:
    """
    Create a directory (and parent directories if needed).

    Args:
        directory_path: Path to the directory to create

    Returns:
        Dictionary with success status and message
    """
    try:
        directory_path = os.path.expanduser(directory_path)
        os.makedirs(directory_path, exist_ok=True)

        return {
            "success": True,
            "message": f"Directory created: {directory_path}",
            "path": directory_path,
        }
    except Exception as e:
        return {
            "success": False,
            "error": f"Error creating directory: {str(e)}",
        }


@function_tool
def delete_file(file_path: str) -> Dict[str, Any]:
    """
    Delete a file.

    Args:
        file_path: Path to the file to delete

    Returns:
        Dictionary with success status and message
    """
    try:
        file_path = os.path.expanduser(file_path)

        if not os.path.exists(file_path):
            return {
                "success": False,
                "error": f"File not found: {file_path}",
            }

        if os.path.isdir(file_path):
            return {
                "success": False,
                "error": f"Path is a directory, use delete_directory instead: {file_path}",
            }

        os.remove(file_path)

        return {
            "success": True,
            "message": f"File deleted: {file_path}",
        }
    except Exception as e:
        return {
            "success": False,
            "error": f"Error deleting file: {str(e)}",
        }


@function_tool
def copy_file(source_path: str, destination_path: str) -> Dict[str, Any]:
    """
    Copy a file from source to destination.

    Args:
        source_path: Path to the source file
        destination_path: Path to the destination

    Returns:
        Dictionary with success status and message
    """
    try:
        source_path = os.path.expanduser(source_path)
        destination_path = os.path.expanduser(destination_path)

        # Create destination directory if needed
        os.makedirs(os.path.dirname(destination_path), exist_ok=True)

        shutil.copy2(source_path, destination_path)

        return {
            "success": True,
            "message": f"File copied from {source_path} to {destination_path}",
            "source": source_path,
            "destination": destination_path,
        }
    except Exception as e:
        return {
            "success": False,
            "error": f"Error copying file: {str(e)}",
        }


@function_tool
def file_exists(file_path: str) -> Dict[str, Any]:
    """
    Check if a file or directory exists.

    Args:
        file_path: Path to check

    Returns:
        Dictionary with existence status and type
    """
    try:
        file_path = os.path.expanduser(file_path)
        exists = os.path.exists(file_path)

        if exists:
            is_file = os.path.isfile(file_path)
            is_dir = os.path.isdir(file_path)

            return {
                "success": True,
                "exists": True,
                "path": file_path,
                "is_file": is_file,
                "is_directory": is_dir,
            }
        else:
            return {
                "success": True,
                "exists": False,
                "path": file_path,
            }
    except Exception as e:
        return {
            "success": False,
            "error": f"Error checking file: {str(e)}",
        }


@function_tool
def get_file_info(file_path: str) -> Dict[str, Any]:
    """
    Get detailed information about a file.

    Args:
        file_path: Path to the file

    Returns:
        Dictionary with file information
    """
    try:
        file_path = os.path.expanduser(file_path)
        path = Path(file_path)

        if not path.exists():
            return {
                "success": False,
                "error": f"File not found: {file_path}",
            }

        stat = path.stat()

        return {
            "success": True,
            "path": str(path),
            "name": path.name,
            "extension": path.suffix,
            "size_bytes": stat.st_size,
            "is_file": path.is_file(),
            "is_directory": path.is_dir(),
            "modified_time": stat.st_mtime,
            "created_time": stat.st_ctime,
        }
    except Exception as e:
        return {
            "success": False,
            "error": f"Error getting file info: {str(e)}",
        }
