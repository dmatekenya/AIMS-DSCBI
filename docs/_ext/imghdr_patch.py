"""
A patch module to handle the imghdr import error in Python 3.13
"""
import sys
import os
import importlib.util
import types

# Check if imghdr module exists
if importlib.util.find_spec('imghdr') is None:
    # Create a mock imghdr module
    mock_imghdr = types.ModuleType('imghdr')
    
    # Add minimal functionality required by Sphinx
    def what(file, h=None):
        """Determine the type of image contained in a file or memory."""
        return None
        
    mock_imghdr.what = what
    mock_imghdr.tests = []
    
    # Add the mock module to sys.modules
    sys.modules['imghdr'] = mock_imghdr
    
    print("Added mock imghdr module for compatibility with Python 3.13")

def setup(app):
    """Setup function for the Sphinx extension."""
    return {
        'version': '0.1',
        'parallel_read_safe': True,
        'parallel_write_safe': True,
    }
