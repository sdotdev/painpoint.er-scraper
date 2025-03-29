"""Console utility functions for PainPoint.er Search."""

import os
import sys
import platform


def clear_console():
    """Clear the console screen in a cross-platform way."""
    # For Windows
    if platform.system() == "Windows":
        os.system('cls')
    # For Unix/Linux/MacOS
    else:
        os.system('clear')


def display_logo():
    """Display the ASCII art logo from the logo.txt file."""
    # Get the directory of this file
    current_dir = os.path.dirname(os.path.abspath(__file__))
    logo_path = os.path.join(current_dir, 'logo.txt')
    
    try:
        with open(logo_path, 'r', encoding='utf-8') as logo_file:
            logo_content = logo_file.read()
            print(logo_content)
    except FileNotFoundError:
        print("Logo file not found. Continuing without logo display.")
    except Exception as e:
        print(f"Error displaying logo: {str(e)}")