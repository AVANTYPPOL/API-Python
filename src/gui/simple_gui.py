#!/usr/bin/env python3
"""
Simple Uber Pricing GUI
Clean implementation that works with the new directory structure
"""

import os
import sys
import tkinter as tk
from tkinter import messagebox

# Add src to Python path
sys.path.insert(0, os.path.join(os.path.dirname(__file__), 'src'))

def main():
    # Check for environment variables
    if 'GOOGLE_MAPS_API_KEY' not in os.environ:
        print("⚠️  Warning: GOOGLE_MAPS_API_KEY not set")
        print("Please set your API keys in environment variables")
        print("Example: set GOOGLE_MAPS_API_KEY=your_key_here")
        
    if 'WEATHER_API_KEY' not in os.environ:
        print("⚠️  Warning: WEATHER_API_KEY not set")
        print("Please set your API keys in environment variables")
        print("Example: set WEATHER_API_KEY=your_key_here")
    
    # Create simple GUI
    root = tk.Tk()
    root.title("🚗 Miami Uber Pricing")
    root.geometry("600x400")
    
    # Header
    header = tk.Label(
        root,
        text="🚗 Miami Uber Pricing Calculator",
        font=("Arial", 20, "bold"),
        bg="#000000",
        fg="white",
        pady=20
    )
    header.pack(fill=tk.X)
    
    # Main content
    main_frame = tk.Frame(root, padx=20, pady=20)
    main_frame.pack(fill=tk.BOTH, expand=True)
    
    # Instructions
    instructions = tk.Label(
        main_frame,
        text="Welcome to the Miami Uber Pricing Calculator!\n\n"
             "To use this application, you need to:\n"
             "1. Set your Google Maps API key: GOOGLE_MAPS_API_KEY\n"
             "2. Set your Weather API key: WEATHER_API_KEY\n\n"
             "Then you can calculate prices for Miami Uber rides.",
        font=("Arial", 12),
        justify=tk.LEFT,
        wraplength=500
    )
    instructions.pack(pady=20)
    
    # Status
    if 'GOOGLE_MAPS_API_KEY' in os.environ and 'WEATHER_API_KEY' in os.environ:
        status_text = "✅ API keys are set! Ready to calculate prices."
        status_color = "green"
    else:
        status_text = "❌ API keys not set. Please configure environment variables."
        status_color = "red"
    
    status = tk.Label(
        main_frame,
        text=status_text,
        font=("Arial", 12, "bold"),
        fg=status_color
    )
    status.pack(pady=10)
    
    # Simple input fields
    tk.Label(main_frame, text="Pickup Address:", font=("Arial", 10)).pack(anchor=tk.W, pady=(20, 5))
    pickup_entry = tk.Entry(main_frame, width=60, font=("Arial", 10))
    pickup_entry.pack(pady=(0, 10))
    pickup_entry.insert(0, "Miami International Airport")
    
    tk.Label(main_frame, text="Dropoff Address:", font=("Arial", 10)).pack(anchor=tk.W, pady=(0, 5))
    dropoff_entry = tk.Entry(main_frame, width=60, font=("Arial", 10))
    dropoff_entry.pack(pady=(0, 20))
    dropoff_entry.insert(0, "South Beach, Miami")
    
    def calculate_price():
        pickup = pickup_entry.get().strip()
        dropoff = dropoff_entry.get().strip()
        
        if not pickup or not dropoff:
            messagebox.showwarning("Input Error", "Please enter both addresses")
            return
        
        if 'GOOGLE_MAPS_API_KEY' not in os.environ:
            messagebox.showerror("Configuration Error", "Google Maps API key not set")
            return
            
        messagebox.showinfo("Demo Mode", 
                          f"This would calculate price from:\n{pickup}\nto:\n{dropoff}\n\n"
                          f"Full functionality requires the complete model files.")
    
    # Calculate button
    calc_btn = tk.Button(
        main_frame,
        text="Calculate Price",
        command=calculate_price,
        bg="#276EF1",
        fg="white",
        font=("Arial", 12, "bold"),
        padx=20,
        pady=10
    )
    calc_btn.pack(pady=10)
    
    # Footer
    footer = tk.Label(
        root,
        text="Miami Uber Pricing Calculator - Secure & Professional",
        font=("Arial", 8),
        bg="#f0f0f0",
        pady=5
    )
    footer.pack(side=tk.BOTTOM, fill=tk.X)
    
    root.mainloop()

if __name__ == "__main__":
    main() 