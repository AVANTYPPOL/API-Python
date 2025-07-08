"""
Miami-Only Uber Pricing GUI
===========================

GUI application using the pure Miami-only model for maximum accuracy.
No NYC influence - just Miami market pricing.

Author: AI Assistant
Date: 2024
"""

import tkinter as tk
from tkinter import ttk, messagebox
import requests
import json
import os
import sys
sys.path.append(os.path.join(os.path.dirname(__file__), '..'))

from models.miami_only_model import MiamiOnlyModel
import math
import threading
import time
import datetime

# API Keys from environment variables
GOOGLE_MAPS_API_KEY = os.environ.get('GOOGLE_MAPS_API_KEY')
WEATHER_API_KEY = os.environ.get('WEATHER_API_KEY')

if not GOOGLE_MAPS_API_KEY:
    raise ValueError("GOOGLE_MAPS_API_KEY environment variable not set")
if not WEATHER_API_KEY:
    raise ValueError("WEATHER_API_KEY environment variable not set")

class AutocompleteEntry(tk.Entry):
    def __init__(self, master, **kwargs):
        super().__init__(master, **kwargs)
        self.suggestions = []
        self.dropdown = None
        
        # Bind events
        self.bind('<KeyRelease>', self.on_key_release)
        self.bind('<FocusOut>', self.on_focus_out)
        self.bind('<FocusIn>', self.on_focus_in)
        
    def on_key_release(self, event):
        """Handle key release events"""
        if event.keysym in ['Up', 'Down', 'Left', 'Right', 'Tab']:
            return
            
        text = self.get()
        if len(text) >= 3:
            self.get_suggestions(text)
        else:
            self.hide_dropdown()
    
    def get_suggestions(self, text):
        """Get address suggestions from Google Places API"""
        try:
            url = "https://maps.googleapis.com/maps/api/place/autocomplete/json"
            params = {
                'input': text,
                'key': GOOGLE_MAPS_API_KEY,
                'components': 'country:us',
                'location': '25.7617,-80.1918',  # Miami center
                'radius': 50000,  # 50km radius
                'types': 'establishment|geocode'
            }
            
            response = requests.get(url, params=params, timeout=3)
            if response.status_code == 200:
                data = response.json()
                self.suggestions = [pred['description'] for pred in data.get('predictions', [])][:5]
                if self.suggestions:
                    self.show_dropdown()
                else:
                    self.hide_dropdown()
        except:
            self.hide_dropdown()
    
    def show_dropdown(self):
        """Show the dropdown with suggestions"""
        if not self.suggestions:
            return
            
        # Create dropdown if it doesn't exist
        if not self.dropdown:
            self.dropdown = tk.Toplevel(self.master)
            self.dropdown.wm_overrideredirect(True)
            self.dropdown.configure(bg='white', highlightbackground='#cccccc', highlightthickness=1)
            
        # Make sure dropdown is visible
        self.dropdown.deiconify()
        
        # Position dropdown below entry
        x = self.winfo_rootx()
        y = self.winfo_rooty() + self.winfo_height()
        self.dropdown.geometry(f"+{x}+{y}")
        
        # Clear previous suggestions
        for widget in self.dropdown.winfo_children():
            widget.destroy()
            
        # Add suggestions
        for i, suggestion in enumerate(self.suggestions):
            label = tk.Label(
                self.dropdown,
                text=suggestion,
                bg='white',
                anchor='w',
                padx=5,
                pady=5,
                cursor='hand2'
            )
            label.pack(fill=tk.X)
            label.bind('<Enter>', lambda e, l=label: l.configure(bg='#e0e0e0'))
            label.bind('<Leave>', lambda e, l=label: l.configure(bg='white'))
            label.bind('<Button-1>', lambda e, s=suggestion: self.select_suggestion(s))
            
        # Update dropdown size
        self.dropdown.update_idletasks()
        width = self.winfo_width()
        self.dropdown.configure(width=width)
        
        # Bring dropdown to front
        self.dropdown.lift()
        
    def hide_dropdown(self, event=None):
        """Hide the dropdown"""
        if self.dropdown:
            self.dropdown.withdraw()
            # Don't destroy, just hide so it can be reused
            
    def select_suggestion(self, suggestion):
        """Select a suggestion from dropdown"""
        self.delete(0, tk.END)
        self.insert(0, suggestion)
        self.hide_dropdown()
        
    def on_focus_out(self, event=None):
        """Handle focus out event"""
        # Delay hiding to allow clicking on suggestions
        self.after(200, self.hide_dropdown)
        
    def on_focus_in(self, event=None):
        """Handle focus in event"""
        # Reset the autocomplete functionality
        pass
        
    def clear_entry(self):
        """Clear the entry and reset autocomplete"""
        self.delete(0, tk.END)
        self.suggestions = []
        self.hide_dropdown()

class MiamiUberGUI:
    def __init__(self, root):
        self.root = root
        self.root.title("🏖️ Miami Uber Pricing - Pure Local Model")
        self.root.geometry("800x900")
        self.root.configure(bg='#f0f8ff')
        
        # Load Miami-only model
        print("🏖️ Loading Miami-Only Model...")
        self.model = MiamiOnlyModel()
        try:
            self.model.load_model('miami_only_model.pkl')
            print("✅ Miami-only model loaded successfully!")
        except Exception as e:
            print(f"❌ Error loading Miami-only model: {e}")
            messagebox.showerror("Error", "Could not load Miami-only model!")
            return
        
        self.setup_ui()
        
    def setup_ui(self):
        """Setup the user interface"""
        
        # Header
        header_frame = tk.Frame(self.root, bg='#2c3e50', height=100)
        header_frame.pack(fill=tk.X, pady=(0, 20))
        header_frame.pack_propagate(False)
        
        title_label = tk.Label(
            header_frame, 
            text="🏖️ MIAMI UBER PRICING",
            font=('Arial', 24, 'bold'),
            fg='white',
            bg='#2c3e50'
        )
        title_label.pack(expand=True)
        
        subtitle_label = tk.Label(
            header_frame,
            text="Pure Miami Model • No NYC Bias • 1,966 Local Rides",
            font=('Arial', 12),
            fg='#ecf0f1',
            bg='#2c3e50'
        )
        subtitle_label.pack()
        
        # Main container
        main_frame = tk.Frame(self.root, bg='#f0f8ff')
        main_frame.pack(fill=tk.BOTH, expand=True, padx=20)
        
        # Address input section
        address_frame = tk.LabelFrame(
            main_frame, 
            text="📍 Trip Details", 
            font=('Arial', 14, 'bold'),
            bg='#f0f8ff',
            fg='#2c3e50',
            padx=10,
            pady=10
        )
        address_frame.pack(fill=tk.X, pady=(0, 20))
        
        # Pickup address
        tk.Label(address_frame, text="Pickup Address:", font=('Arial', 12), bg='#f0f8ff').pack(anchor='w')
        self.pickup_entry = AutocompleteEntry(address_frame, font=('Arial', 12), width=60)
        self.pickup_entry.pack(fill=tk.X, pady=(5, 10))
        
        # Dropoff address
        tk.Label(address_frame, text="Dropoff Address:", font=('Arial', 12), bg='#f0f8ff').pack(anchor='w')
        self.dropoff_entry = AutocompleteEntry(address_frame, font=('Arial', 12), width=60)
        self.dropoff_entry.pack(fill=tk.X, pady=(5, 15))
        
        # Calculate button
        calc_button = tk.Button(
            address_frame,
            text="🚗 Calculate Miami Prices",
            font=('Arial', 14, 'bold'),
            bg='#3498db',
            fg='white',
            command=self.calculate_prices,
            padx=20,
            pady=10
        )
        calc_button.pack(pady=10)
        
        # Quick routes section
        routes_frame = tk.LabelFrame(
            main_frame,
            text="🚀 Popular Miami Routes",
            font=('Arial', 14, 'bold'),
            bg='#f0f8ff',
            fg='#2c3e50',
            padx=10,
            pady=10
        )
        routes_frame.pack(fill=tk.X, pady=(0, 20))
        
        # Quick route buttons
        routes = [
            ("🛫 Airport → South Beach", "Miami International Airport, Miami, FL", "Ocean Drive, South Beach, Miami Beach, FL"),
            ("🏢 Downtown → Wynwood", "Bayside Marketplace, Miami, FL", "Wynwood Walls, Miami, FL"),
            ("🏖️ Brickell → Coral Gables", "Brickell City Centre, Miami, FL", "Miracle Mile, Coral Gables, FL"),
            ("🌴 South Beach → Downtown", "Lincoln Road, Miami Beach, FL", "Downtown Miami, FL")
        ]
        
        button_frame = tk.Frame(routes_frame, bg='#f0f8ff')
        button_frame.pack(fill=tk.X)
        
        for i, (name, pickup, dropoff) in enumerate(routes):
            row = i // 2
            col = i % 2
            
            btn = tk.Button(
                button_frame,
                text=name,
                font=('Arial', 10),
                bg='#e74c3c',
                fg='white',
                command=lambda p=pickup, d=dropoff: self.set_route(p, d),
                padx=10,
                pady=5
            )
            btn.grid(row=row, column=col, padx=5, pady=5, sticky='ew')
        
        button_frame.grid_columnconfigure(0, weight=1)
        button_frame.grid_columnconfigure(1, weight=1)
        
        # Results section
        self.results_frame = tk.LabelFrame(
            main_frame,
            text="💰 Miami Pricing Results",
            font=('Arial', 14, 'bold'),
            bg='#f0f8ff',
            fg='#2c3e50',
            padx=15,
            pady=15
        )
        self.results_frame.pack(fill=tk.BOTH, expand=True)
        
        # Initially hidden
        self.results_frame.pack_forget()
        
    def set_route(self, pickup, dropoff):
        """Set a popular route"""
        # Clear and set pickup
        self.pickup_entry.clear_entry()
        self.pickup_entry.insert(0, pickup)
        
        # Clear and set dropoff
        self.dropoff_entry.clear_entry()
        self.dropoff_entry.insert(0, dropoff)
        
        self.calculate_prices()
    
    def geocode_address(self, address):
        """Convert address to coordinates"""
        try:
            url = "https://maps.googleapis.com/maps/api/geocode/json"
            params = {
                'address': address,
                'key': GOOGLE_MAPS_API_KEY,
                'components': 'country:US'
            }
            
            response = requests.get(url, params=params, timeout=10)
            if response.status_code == 200:
                data = response.json()
                if data['results']:
                    location = data['results'][0]['geometry']['location']
                    return location['lat'], location['lng']
            return None, None
        except Exception as e:
            print(f"Geocoding error: {e}")
            return None, None
    
    def calculate_distance(self, pickup_lat, pickup_lng, dropoff_lat, dropoff_lng):
        """Calculate distance between two points"""
        try:
            url = "https://maps.googleapis.com/maps/api/distancematrix/json"
            params = {
                'origins': f"{pickup_lat},{pickup_lng}",
                'destinations': f"{dropoff_lat},{dropoff_lng}",
                'key': GOOGLE_MAPS_API_KEY,
                'units': 'metric'
            }
            
            response = requests.get(url, params=params, timeout=10)
            if response.status_code == 200:
                data = response.json()
                if data['rows'] and data['rows'][0]['elements']:
                    element = data['rows'][0]['elements'][0]
                    if element['status'] == 'OK':
                        distance_m = element['distance']['value']
                        return distance_m / 1000  # Convert to km
            
            # Fallback to haversine formula
            return self.haversine_distance(pickup_lat, pickup_lng, dropoff_lat, dropoff_lng)
        except:
            return self.haversine_distance(pickup_lat, pickup_lng, dropoff_lat, dropoff_lng)
    
    def haversine_distance(self, lat1, lon1, lat2, lon2):
        """Calculate distance using haversine formula"""
        R = 6371  # Earth's radius in km
        
        lat1, lon1, lat2, lon2 = map(math.radians, [lat1, lon1, lat2, lon2])
        dlat = lat2 - lat1
        dlon = lon2 - lon1
        
        a = math.sin(dlat/2)**2 + math.cos(lat1) * math.cos(lat2) * math.sin(dlon/2)**2
        c = 2 * math.asin(math.sqrt(a))
        
        return R * c
    
    def get_current_conditions(self):
        """Get current traffic and weather conditions"""
        # For now, return default conditions
        # In a real app, you'd call actual APIs
        return 'moderate', 'clear'
    
    def calculate_prices(self):
        """Calculate prices using Miami-only model"""
        pickup_address = self.pickup_entry.get().strip()
        dropoff_address = self.dropoff_entry.get().strip()
        
        if not pickup_address or not dropoff_address:
            messagebox.showwarning("Input Error", "Please enter both pickup and dropoff addresses")
            return
        
        # Show loading
        self.show_loading()
        
        # Run calculation in thread to prevent GUI freezing
        def calculate():
            try:
                # Geocode addresses
                pickup_lat, pickup_lng = self.geocode_address(pickup_address)
                dropoff_lat, dropoff_lng = self.geocode_address(dropoff_address)
                
                if pickup_lat is None or dropoff_lat is None:
                    self.root.after(0, lambda: messagebox.showerror("Error", "Could not find one or both addresses"))
                    self.root.after(0, self.hide_loading)
                    return
                
                # Calculate distance
                distance_km = self.calculate_distance(pickup_lat, pickup_lng, dropoff_lat, dropoff_lng)
                
                # Get current conditions
                traffic_level, weather_condition = self.get_current_conditions()
                
                # Get current time
                now = datetime.datetime.now()
                hour_of_day = now.hour
                day_of_week = now.weekday()
                
                # Make prediction using Miami-only model
                predictions = self.model.predict(
                    pickup_lat=pickup_lat,
                    pickup_lng=pickup_lng,
                    dropoff_lat=dropoff_lat,
                    dropoff_lng=dropoff_lng,
                    distance_km=distance_km,
                    hour_of_day=hour_of_day,
                    day_of_week=day_of_week,
                    surge_multiplier=1.0,
                    traffic_level=traffic_level,
                    weather_condition=weather_condition
                )
                
                # Update UI in main thread
                self.root.after(0, lambda: self.show_results(predictions, distance_km, traffic_level, weather_condition))
                
            except Exception as e:
                print(f"Calculation error: {e}")
                self.root.after(0, lambda: messagebox.showerror("Error", f"Calculation failed: {str(e)}"))
                self.root.after(0, self.hide_loading)
        
        threading.Thread(target=calculate, daemon=True).start()
    
    def show_loading(self):
        """Show loading state"""
        # Clear previous results
        for widget in self.results_frame.winfo_children():
            widget.destroy()
        
        self.results_frame.pack(fill=tk.BOTH, expand=True)
        
        loading_label = tk.Label(
            self.results_frame,
            text="🔄 Calculating Miami prices...",
            font=('Arial', 16),
            bg='#f0f8ff',
            fg='#3498db'
        )
        loading_label.pack(expand=True)
    
    def hide_loading(self):
        """Hide loading state"""
        self.results_frame.pack_forget()
    
    def show_results(self, predictions, distance_km, traffic_level, weather_condition):
        """Display pricing results"""
        # Clear loading
        for widget in self.results_frame.winfo_children():
            widget.destroy()
        
        # Trip info
        info_frame = tk.Frame(self.results_frame, bg='#f0f8ff')
        info_frame.pack(fill=tk.X, pady=(0, 20))
        
        info_text = f"📏 Distance: {distance_km:.1f} km • 🚦 Traffic: {traffic_level.title()} • 🌤️ Weather: {weather_condition.title()}"
        tk.Label(
            info_frame,
            text=info_text,
            font=('Arial', 12),
            bg='#f0f8ff',
            fg='#7f8c8d'
        ).pack()
        
        # Service prices
        services = [
            ('uberx', 'UberX', '#1abc9c', 'Standard ride'),
            ('uber_xl', 'UberXL', '#3498db', 'Larger vehicle, up to 6 people'),
            ('uber_premier', 'Uber Premier', '#9b59b6', 'Premium ride'),
            ('premier_suv', 'Premier SUV', '#e74c3c', 'Luxury SUV')
        ]
        
        for service_key, service_name, color, description in services:
            if service_key in predictions:
                price = predictions[service_key]
                
                service_frame = tk.Frame(self.results_frame, bg='white', relief='raised', bd=2)
                service_frame.pack(fill=tk.X, pady=5, padx=10)
                
                # Service header
                header_frame = tk.Frame(service_frame, bg=color, height=40)
                header_frame.pack(fill=tk.X)
                header_frame.pack_propagate(False)
                
                tk.Label(
                    header_frame,
                    text=service_name,
                    font=('Arial', 14, 'bold'),
                    fg='white',
                    bg=color
                ).pack(side=tk.LEFT, padx=15, pady=8)
                
                tk.Label(
                    header_frame,
                    text=f"${price:.2f}",
                    font=('Arial', 18, 'bold'),
                    fg='white',
                    bg=color
                ).pack(side=tk.RIGHT, padx=15, pady=8)
                
                # Service description
                tk.Label(
                    service_frame,
                    text=description,
                    font=('Arial', 10),
                    fg='#7f8c8d',
                    bg='white'
                ).pack(pady=(5, 10), padx=15, anchor='w')
        
        # Model info
        model_info = tk.Label(
            self.results_frame,
            text="🏖️ Powered by Miami-Only Model • Trained on 1,966 local rides",
            font=('Arial', 10, 'italic'),
            bg='#f0f8ff',
            fg='#95a5a6'
        )
        model_info.pack(pady=(20, 0))

def main():
    """Main function"""
    print("🏖️ Starting Miami Uber Pricing GUI...")
    
    root = tk.Tk()
    app = MiamiUberGUI(root)
    
    print("✅ GUI ready!")
    root.mainloop()

if __name__ == "__main__":
    main() 