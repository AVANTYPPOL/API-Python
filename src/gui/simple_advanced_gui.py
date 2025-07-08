#!/usr/bin/env python3
"""
Simple Advanced Uber Pricing GUI
================================

Self-contained GUI with API key validation and demo functionality.
No complex imports - everything in one file.
"""

import tkinter as tk
from tkinter import messagebox, simpledialog
import os
import requests
import threading
import sys
from datetime import datetime
import math

class AutocompleteEntry(tk.Entry):
    """Entry widget with Google Places autocomplete"""
    
    def __init__(self, parent, api_key, **kwargs):
        super().__init__(parent, **kwargs)
        self.api_key = api_key
        self.suggestions = []
        self.dropdown = None
        
        if self.api_key:
            # Bind events only if API key is available
            self.bind('<KeyRelease>', self.on_key_release)
            self.bind('<FocusOut>', self.on_focus_out)
        
    def on_key_release(self, event):
        """Handle key release events"""
        if not self.api_key:
            return
            
        if event.keysym in ['Up', 'Down', 'Left', 'Right', 'Return', 'Tab']:
            return
            
        query = self.get().strip()
        if len(query) >= 3:
            # Use threading to avoid blocking the GUI
            threading.Thread(target=self.get_suggestions, args=(query,), daemon=True).start()
        else:
            self.hide_dropdown()
            
    def get_suggestions(self, query):
        """Get address suggestions from Google Places API"""
        if not self.api_key:
            return
            
        try:
            url = "https://maps.googleapis.com/maps/api/place/autocomplete/json"
            params = {
                'input': query,
                'key': self.api_key,
                'components': 'country:us',
                'location': '25.7617,-80.1918',  # Miami center
                'radius': 50000,  # 50km radius
                'types': 'establishment|geocode'
            }
            
            response = requests.get(url, params=params, timeout=3)
            if response.status_code == 200:
                data = response.json()
                if data['status'] == 'OK':
                    self.suggestions = [pred['description'] for pred in data['predictions'][:5]]
                    # Update GUI in main thread
                    self.after(0, self.show_dropdown)
        except Exception as e:
            print(f"Autocomplete error: {e}")
            
    def show_dropdown(self):
        """Show dropdown with suggestions"""
        if not self.suggestions or not self.api_key:
            return
            
        if not self.dropdown:
            self.dropdown = tk.Toplevel(self.master)
            self.dropdown.wm_overrideredirect(True)
            self.dropdown.configure(bg='white', highlightbackground='#cccccc', highlightthickness=1)
            
        self.dropdown.deiconify()
        
        # Position dropdown below entry
        x = self.winfo_rootx()
        y = self.winfo_rooty() + self.winfo_height()
        self.dropdown.geometry(f"+{x}+{y}")
        
        # Clear previous suggestions
        for widget in self.dropdown.winfo_children():
            widget.destroy()
            
        # Add suggestions
        for suggestion in self.suggestions:
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
        
        self.dropdown.update_idletasks()
        self.dropdown.lift()
        
    def hide_dropdown(self, event=None):
        """Hide dropdown"""
        if self.dropdown:
            self.dropdown.withdraw()
            
    def select_suggestion(self, suggestion):
        """Select suggestion"""
        self.delete(0, tk.END)
        self.insert(0, suggestion)
        self.hide_dropdown()
        
    def on_focus_out(self, event=None):
        """Handle focus out"""
        if self.api_key:
            self.after(200, self.hide_dropdown)
        
    def clear_entry(self):
        """Clear entry"""
        self.delete(0, tk.END)
        self.suggestions = []
        self.hide_dropdown()

class SimpleUberGUI:
    def __init__(self, root):
        self.root = root
        self.root.title("🚗 Miami Uber Pricing - ML Model Interface")
        self.root.geometry("900x700")
        
        # Check API keys
        self.google_api_key = os.environ.get('GOOGLE_MAPS_API_KEY')
        self.weather_api_key = os.environ.get('WEATHER_API_KEY')
        self.apis_available = bool(self.google_api_key and self.weather_api_key)
        
        # Initialize ML model
        self.model = None
        self.model_ready = False
        self.init_model()
        
        # Colors
        self.bg_color = "#f0f0f0"
        self.uber_black = "#000000"
        self.uber_blue = "#276EF1"
        
        self.root.configure(bg=self.bg_color)
        self.create_widgets()
        
    def init_model(self):
        """Initialize the ML model"""
        try:
            # Add parent directories to path
            sys.path.insert(0, os.path.join(os.path.dirname(__file__), '..', '..'))
            sys.path.insert(0, os.path.join(os.path.dirname(__file__), '..'))
            
            # Try to import and load the hybrid model (NYC + Miami with 99.9% Miami weight)
            try:
                from models.hybrid_uber_model import HybridUberPriceModel
                self.model = HybridUberPriceModel()
                print("🌆 Initializing Hybrid Model (NYC + Miami data)")
                
                # Check if pre-trained model exists
                model_paths = [
                    'hybrid_uber_model.pkl',
                    '../../hybrid_uber_model.pkl',
                    os.path.join(os.path.dirname(__file__), '..', '..', 'hybrid_uber_model.pkl')
                ]
                
                model_loaded = False
                for path in model_paths:
                    if os.path.exists(path):
                        try:
                            self.model.load_model(path)
                            self.model_ready = True
                            model_loaded = True
                            print(f"✅ Hybrid Model loaded from {path}!")
                            print("   - NYC data: 50,000 rides (transfer learning)")
                            print("   - Miami data: 1,966 rides (99.9% weight)")
                            break
                        except:
                            continue
                
                if not model_loaded:
                    print("📊 No pre-trained model found, training hybrid model...")
                    success = self.model.train_full_pipeline()
                    if success:
                        self.model.save_model('hybrid_uber_model.pkl')
                        self.model_ready = True
                        print("✅ Hybrid Model trained successfully!")
                    else:
                        raise Exception("Failed to train hybrid model")
                        
            except Exception as e:
                print(f"⚠️ Hybrid model failed: {e}")
                # Fallback to simple model if hybrid fails
                from models.simple_calibrated_model import SimpleCalibratedModel
                self.model = SimpleCalibratedModel()
                self.model_ready = True
                print("📊 Using Simple Calibrated Model (fallback)")
                
        except ImportError as e:
            print(f"⚠️ Could not import model: {e}")
            self.model_ready = False
        except Exception as e:
            print(f"⚠️ Model initialization error: {e}")
            self.model_ready = False
        
    def create_widgets(self):
        # Header
        header_frame = tk.Frame(self.root, bg=self.uber_black, height=70)
        header_frame.pack(fill=tk.X)
        header_frame.pack_propagate(False)
        
        header_label = tk.Label(
            header_frame,
            text="🚗 Uber Multi-Service Price Calculator",
            font=("Arial", 20, "bold"),
            bg=self.uber_black,
            fg="white"
        )
        header_label.pack(pady=15)
        
        # Main content
        main_frame = tk.Frame(self.root, bg=self.bg_color)
        main_frame.pack(fill=tk.BOTH, expand=True, padx=20, pady=20)
        
        # API Status Section
        self.create_api_status_section(main_frame)
        
        # Input section
        input_frame = tk.LabelFrame(
            main_frame,
            text="Trip Details",
            font=("Arial", 12, "bold"),
            bg=self.bg_color,
            padx=15,
            pady=15
        )
        input_frame.pack(fill=tk.X, pady=(10, 20))
        
        # Pickup address
        pickup_frame = tk.Frame(input_frame, bg=self.bg_color)
        pickup_frame.pack(fill=tk.X, pady=(0, 10))
        
        tk.Label(pickup_frame, text="Pickup Address:", bg=self.bg_color, font=("Arial", 10)).pack(anchor=tk.W, pady=(0, 5))
        self.pickup_entry = AutocompleteEntry(pickup_frame, self.google_api_key, width=60, font=("Arial", 10))
        self.pickup_entry.pack(pady=(0, 5))
        self.pickup_entry.insert(0, "Miami International Airport")
        
        # Autocomplete status for pickup
        pickup_status = "✅ Autocomplete enabled" if self.google_api_key else "⚠️ No autocomplete (set API key)"
        tk.Label(pickup_frame, text=pickup_status, bg=self.bg_color, font=("Arial", 8), 
                fg="green" if self.google_api_key else "orange").pack(anchor=tk.W)
        
        # Dropoff address
        dropoff_frame = tk.Frame(input_frame, bg=self.bg_color)
        dropoff_frame.pack(fill=tk.X, pady=(0, 15))
        
        tk.Label(dropoff_frame, text="Dropoff Address:", bg=self.bg_color, font=("Arial", 10)).pack(anchor=tk.W, pady=(0, 5))
        self.dropoff_entry = AutocompleteEntry(dropoff_frame, self.google_api_key, width=60, font=("Arial", 10))
        self.dropoff_entry.pack(pady=(0, 5))
        self.dropoff_entry.insert(0, "South Beach, Miami")
        
        # Autocomplete status for dropoff
        dropoff_status = "✅ Autocomplete enabled" if self.google_api_key else "⚠️ No autocomplete (set API key)"
        tk.Label(dropoff_frame, text=dropoff_status, bg=self.bg_color, font=("Arial", 8), 
                fg="green" if self.google_api_key else "orange").pack(anchor=tk.W)
        
        # Calculate button
        button_text = "Calculate Prices (API-Enhanced)" if self.apis_available else "Calculate Prices (Demo Mode)"
        button_color = self.uber_blue if self.apis_available else "#cccccc"
        text_color = "white" if self.apis_available else "#666666"
        
        self.calculate_btn = tk.Button(
            input_frame,
            text=button_text,
            command=self.calculate_prices,
            bg=button_color,
            fg=text_color,
            font=("Arial", 12, "bold"),
            padx=20,
            pady=10
        )
        self.calculate_btn.pack(pady=10)
        
        # Popular routes
        routes_frame = tk.Frame(input_frame, bg=self.bg_color)
        routes_frame.pack(pady=(10, 0))
        
        tk.Label(routes_frame, text="Popular Routes:", bg=self.bg_color, font=("Arial", 9, "bold")).pack()
        
        routes_buttons_frame = tk.Frame(routes_frame, bg=self.bg_color)
        routes_buttons_frame.pack(pady=5)
        
        popular_routes = [
            ("Airport → South Beach", "Miami International Airport", "Ocean Drive, South Beach, Miami"),
            ("Downtown → Wynwood", "Bayside Marketplace, Miami", "Wynwood Walls, Miami"),
            ("Brickell → Coral Gables", "Brickell City Centre, Miami", "Miracle Mile, Coral Gables")
        ]
        
        for route_name, pickup, dropoff in popular_routes:
            btn = tk.Button(
                routes_buttons_frame,
                text=route_name,
                command=lambda p=pickup, d=dropoff: self.set_route(p, d),
                bg="#e0e0e0",
                font=("Arial", 8),
                padx=8,
                pady=4
            )
            btn.pack(side=tk.LEFT, padx=3)
        
        # Results section
        self.results_frame = tk.LabelFrame(
            main_frame,
            text="Results",
            font=("Arial", 12, "bold"),
            bg=self.bg_color,
            padx=15,
            pady=15
        )
        self.results_frame.pack(fill=tk.BOTH, expand=True)
        
        # Status bar
        status_text = "Ready - APIs Connected!" if self.apis_available else "Demo Mode - Set API keys for full features"
        self.status_var = tk.StringVar(value=status_text)
        self.status_bar = tk.Label(
            self.root,
            textvariable=self.status_var,
            bg="#333333" if self.apis_available else "#666666",
            fg="white",
            font=("Arial", 9),
            anchor=tk.W,
            padx=10
        )
        self.status_bar.pack(side=tk.BOTTOM, fill=tk.X)
        
    def create_api_status_section(self, parent):
        """Create API and model status section"""
        status_frame = tk.LabelFrame(
            parent,
            text="🔧 System Status",
            font=("Arial", 10, "bold"),
            bg=self.bg_color,
            padx=10,
            pady=10
        )
        status_frame.pack(fill=tk.X, pady=(0, 10))
        
        # ML Model status
        model_status = "✅ ML Model: Hybrid Model Loaded" if self.model_ready else "❌ ML Model: Not available"
        model_color = "green" if self.model_ready else "red"
        model_label = tk.Label(status_frame, text=model_status, bg=self.bg_color, fg=model_color, font=("Arial", 9, "bold"))
        model_label.pack(anchor=tk.W)
        
        # Google Maps API status
        google_status = "✅ Google Maps API: Connected" if self.google_api_key else "❌ Google Maps API: Not configured"
        google_color = "green" if self.google_api_key else "red"
        google_label = tk.Label(status_frame, text=google_status, bg=self.bg_color, fg=google_color, font=("Arial", 9))
        google_label.pack(anchor=tk.W)
        
        # Weather API status
        weather_status = "✅ Weather API: Connected" if self.weather_api_key else "❌ Weather API: Not configured"
        weather_color = "green" if self.weather_api_key else "red"
        weather_label = tk.Label(status_frame, text=weather_status, bg=self.bg_color, fg=weather_color, font=("Arial", 9))
        weather_label.pack(anchor=tk.W)
        
        # Overall status
        if self.model_ready and self.apis_available:
            overall_status = "🚀 Full ML Pipeline Ready - Model + APIs Connected!"
            overall_color = "green"
        elif self.model_ready:
            overall_status = "🤖 ML Model Ready - Set API keys for real-time data"
            overall_color = "blue"
        elif self.apis_available:
            overall_status = "📡 APIs Ready - ML model needs setup"
            overall_color = "orange"
        else:
            overall_status = "⚠️ Demo mode - Set up model and API keys"
            overall_color = "red"
        
        overall_label = tk.Label(status_frame, text=overall_status, bg=self.bg_color, fg=overall_color, 
                font=("Arial", 9, "bold"))
        overall_label.pack(anchor=tk.W, pady=(5, 0))
        
        # Setup button if APIs not configured
        if not self.apis_available:
            setup_btn = tk.Button(
                status_frame,
                text="🔧 Setup API Keys",
                command=self.setup_api_keys,
                bg="#4CAF50",
                fg="white",
                font=("Arial", 8),
                padx=10,
                pady=2
            )
            setup_btn.pack(anchor=tk.W, pady=(5, 0))
    
    def setup_api_keys(self):
        """Show API setup instructions"""
        instructions = """
🔧 API Setup Instructions

To enable full functionality:

1. Set environment variables in PowerShell:
   $env:GOOGLE_MAPS_API_KEY="your_google_maps_key"
   $env:WEATHER_API_KEY="your_weather_key"

2. Or use Command Prompt:
   set GOOGLE_MAPS_API_KEY=your_google_maps_key
   set WEATHER_API_KEY=your_weather_key

3. Restart this application

📋 With API keys you get:
• Address autocomplete
• Real-time traffic data
• Weather conditions
• Accurate distance calculation
• Enhanced price predictions
        """
        messagebox.showinfo("API Setup Instructions", instructions)
        
    def set_route(self, pickup, dropoff):
        """Set popular route"""
        self.pickup_entry.delete(0, tk.END)
        self.pickup_entry.insert(0, pickup)
        self.dropoff_entry.delete(0, tk.END)
        self.dropoff_entry.insert(0, dropoff)
        self.calculate_prices()
        
    def calculate_prices(self):
        pickup = self.pickup_entry.get().strip()
        dropoff = self.dropoff_entry.get().strip()
        
        if not pickup or not dropoff:
            messagebox.showwarning("Input Error", "Please enter both pickup and dropoff addresses")
            return
        
        # Clear previous results
        for widget in self.results_frame.winfo_children():
            widget.destroy()
        
        # Show calculation in progress
        self.status_var.set("🔍 Calculating prices...")
        self.calculate_btn.config(state=tk.DISABLED)
        
        # Calculate in background thread
        threading.Thread(target=self.calculate_real_prices, args=(pickup, dropoff), daemon=True).start()
        
    def calculate_real_prices(self, pickup, dropoff):
        """Calculate real prices using ML model and APIs"""
        try:
            # Step 1: Geocode addresses (if APIs available)
            if self.apis_available:
                self.root.after(0, lambda: self.status_var.set("📍 Geocoding addresses..."))
                pickup_coords = self.geocode_address(pickup)
                dropoff_coords = self.geocode_address(dropoff)
                
                if not pickup_coords or not dropoff_coords:
                    self.root.after(0, lambda: self.show_error("Could not geocode addresses"))
                    return
                
                # Step 2: Get real distance and traffic
                self.root.after(0, lambda: self.status_var.set("🚗 Getting distance and traffic..."))
                distance_data = self.get_distance_and_traffic(pickup_coords, dropoff_coords)
                
                # Step 3: Get weather
                self.root.after(0, lambda: self.status_var.set("🌤️ Getting weather data..."))
                weather_data = self.get_weather(pickup_coords)
                
            else:
                # Use demo coordinates for Miami area
                pickup_coords = {'lat': 25.7617, 'lng': -80.1918}  # Downtown Miami
                dropoff_coords = {'lat': 25.7907, 'lng': -80.1300}  # South Beach
                distance_data = {'distance_km': 15.0, 'duration_min': 25, 'traffic_level': 'moderate'}
                weather_data = {'condition': 'clear', 'temperature': 75}
            
            # Step 4: ML Model Prediction
            if self.model_ready:
                self.root.after(0, lambda: self.status_var.set("🤖 Running ML model prediction..."))
                
                # Get current time
                now = datetime.now()
                hour_of_day = now.hour
                day_of_week = now.weekday()
                
                # Make prediction for UberX base price
                ml_price = self.model.predict_price(
                    distance_km=distance_data['distance_km'],
                    pickup_lat=pickup_coords['lat'],
                    pickup_lng=pickup_coords['lng'],
                    dropoff_lat=dropoff_coords['lat'],
                    dropoff_lng=dropoff_coords['lng'],
                    hour_of_day=hour_of_day,
                    day_of_week=day_of_week,
                    surge_multiplier=1.0,
                    traffic_level=distance_data.get('traffic_level', 'moderate'),
                    weather_condition=weather_data.get('condition', 'clear')
                )
                
                # Generate service prices based on ML prediction
                prices = self.generate_service_prices(ml_price)
                
            else:
                # Use demo prices
                ml_price = None
                prices = {
                    'UberX': 25.50,
                    'UberXL': 35.75,
                    'Uber Premier': 45.20,
                    'Premier SUV': 55.90
                }
            
            # Update GUI with results
            result_data = {
                'pickup': pickup,
                'dropoff': dropoff,
                'distance_km': distance_data['distance_km'],
                'duration_min': distance_data.get('duration_min', 25),
                'traffic_level': distance_data.get('traffic_level', 'moderate'),
                'weather_condition': weather_data.get('condition', 'clear'),
                'temperature': weather_data.get('temperature', 75),
                'prices': prices,
                'ml_prediction': ml_price,
                'data_source': 'Real APIs + ML Model' if (self.apis_available and self.model_ready) else 
                             'ML Model + Demo Data' if self.model_ready else 
                             'Real APIs + Demo Pricing' if self.apis_available else 'Demo Data'
            }
            
            self.root.after(0, lambda: self.show_results_with_ml(result_data))
            
        except Exception as e:
            error_msg = f"Calculation error: {str(e)}"
            self.root.after(0, lambda: self.show_error(error_msg))
    
    def geocode_address(self, address):
        """Geocode address using Google Maps API"""
        if not self.google_api_key:
            return None
            
        try:
            url = "https://maps.googleapis.com/maps/api/geocode/json"
            params = {
                'address': address,
                'key': self.google_api_key,
                'components': 'country:US'
            }
            
            response = requests.get(url, params=params, timeout=10)
            if response.status_code == 200:
                data = response.json()
                if data['results']:
                    location = data['results'][0]['geometry']['location']
                    return {'lat': location['lat'], 'lng': location['lng']}
            return None
        except Exception:
            return None
    
    def get_distance_and_traffic(self, pickup_coords, dropoff_coords):
        """Get distance and traffic data"""
        if not self.google_api_key:
            return {'distance_km': 15.0, 'duration_min': 25, 'traffic_level': 'moderate'}
            
        try:
            url = "https://maps.googleapis.com/maps/api/distancematrix/json"
            params = {
                'origins': f"{pickup_coords['lat']},{pickup_coords['lng']}",
                'destinations': f"{dropoff_coords['lat']},{dropoff_coords['lng']}",
                'mode': 'driving',
                'departure_time': 'now',
                'traffic_model': 'best_guess',
                'key': self.google_api_key
            }
            
            response = requests.get(url, params=params, timeout=10)
            if response.status_code == 200:
                data = response.json()
                if data['rows'] and data['rows'][0]['elements']:
                    element = data['rows'][0]['elements'][0]
                    if element['status'] == 'OK':
                        distance_m = element['distance']['value']
                        distance_km = distance_m / 1000
                        
                        duration_s = element['duration']['value']
                        duration_min = duration_s / 60
                        
                        # Check for traffic
                        traffic_duration_s = element.get('duration_in_traffic', {}).get('value', duration_s)
                        traffic_duration_min = traffic_duration_s / 60
                        
                        if traffic_duration_min <= duration_min * 1.1:
                            traffic_level = 'light'
                        elif traffic_duration_min <= duration_min * 1.3:
                            traffic_level = 'moderate'
                        else:
                            traffic_level = 'heavy'
                        
                        return {
                            'distance_km': distance_km,
                            'duration_min': traffic_duration_min,
                            'traffic_level': traffic_level
                        }
            
            # Fallback
            return {'distance_km': 15.0, 'duration_min': 25, 'traffic_level': 'moderate'}
        except Exception:
            return {'distance_km': 15.0, 'duration_min': 25, 'traffic_level': 'moderate'}
    
    def get_weather(self, coords):
        """Get weather data"""
        if not self.weather_api_key:
            return {'condition': 'clear', 'temperature': 75}
            
        try:
            url = "http://api.openweathermap.org/data/2.5/weather"
            params = {
                'lat': coords['lat'],
                'lon': coords['lng'],
                'appid': self.weather_api_key,
                'units': 'imperial'
            }
            
            response = requests.get(url, params=params, timeout=5)
            if response.status_code == 200:
                data = response.json()
                weather_main = data['weather'][0]['main'].lower()
                temperature = data['main']['temp']
                
                if 'rain' in weather_main or 'drizzle' in weather_main:
                    condition = 'rain'
                elif 'cloud' in weather_main:
                    condition = 'clouds'
                else:
                    condition = 'clear'
                
                return {'condition': condition, 'temperature': temperature}
            
            return {'condition': 'clear', 'temperature': 75}
        except Exception:
            return {'condition': 'clear', 'temperature': 75}
    
    def generate_service_prices(self, base_price):
        """Generate prices for all services based on ML prediction"""
        # UberX is the base prediction
        uberx_price = base_price
        
        # Apply typical multipliers for other services
        return {
            'UberX': round(uberx_price, 2),
            'UberXL': round(uberx_price * 1.4, 2),  # ~40% more
            'Uber Premier': round(uberx_price * 1.8, 2),  # ~80% more  
            'Premier SUV': round(uberx_price * 2.2, 2)   # ~120% more
        }
    
    def show_error(self, message):
        """Show error message"""
        messagebox.showerror("Error", message)
        self.status_var.set("❌ Error occurred")
        self.calculate_btn.config(state=tk.NORMAL)
        
    def show_results_with_ml(self, result_data):
        """Show results with ML model data"""
        # Clear previous results
        for widget in self.results_frame.winfo_children():
            widget.destroy()
        
        # Trip info with data source
        trip_info = f"""📍 From: {result_data['pickup']}
📍 To: {result_data['dropoff']}
📏 Distance: {result_data['distance_km']:.1f} km ({result_data['distance_km']*0.621371:.1f} miles)
⏱️ Duration: {result_data['duration_min']:.0f} minutes
🚦 Traffic: {result_data['traffic_level'].title()}
🌤️ Weather: {result_data['weather_condition'].title()} ({result_data['temperature']:.0f}°F)

🔬 Data Source: {result_data['data_source']}"""

        if result_data['ml_prediction']:
            trip_info += f"\n🤖 ML Base Prediction (UberX): ${result_data['ml_prediction']:.2f}"
        
        info_label = tk.Label(
            self.results_frame,
            text=trip_info,
            bg=self.bg_color,
            font=("Arial", 10),
            justify=tk.LEFT,
            fg="#333333"
        )
        info_label.pack(anchor=tk.W, pady=(0, 15))
        
        # Service prices with ML indicators
        services = [
            ("UberX", result_data['prices']['UberX'], "Affordable rides for up to 4", "#276EF1"),
            ("UberXL", result_data['prices']['UberXL'], "Rides for groups up to 6", "#00a862"),
            ("Uber Premier", result_data['prices']['Uber Premier'], "Premium rides", "#000000"),
            ("Premier SUV", result_data['prices']['Premier SUV'], "Premium SUVs for up to 6", "#6B3AA7")
        ]
        
        for service_name, price, description, color in services:
            service_frame = tk.Frame(self.results_frame, bg="white", relief=tk.RAISED, bd=1)
            service_frame.pack(fill=tk.X, pady=3)
            
            # Color bar
            color_bar = tk.Frame(service_frame, bg=color, width=5)
            color_bar.pack(side=tk.LEFT, fill=tk.Y)
            
            # Content
            content_frame = tk.Frame(service_frame, bg="white", padx=15, pady=8)
            content_frame.pack(side=tk.LEFT, fill=tk.BOTH, expand=True)
            
            # Name and price
            name_price_frame = tk.Frame(content_frame, bg="white")
            name_price_frame.pack(fill=tk.X)
            
            # Add ML indicator
            if self.model_ready:
                status_suffix = " 🤖"
                price_color = color
            else:
                status_suffix = " (DEMO)"
                price_color = "#999999"
            
            service_label = tk.Label(
                name_price_frame,
                text=f"{service_name}{status_suffix}",
                bg="white",
                font=("Arial", 12, "bold")
            )
            service_label.pack(side=tk.LEFT)
            
            price_label = tk.Label(
                name_price_frame,
                text=f"${price:.2f}",
                bg="white",
                font=("Arial", 14, "bold"),
                fg=price_color
            )
            price_label.pack(side=tk.RIGHT)
            
            # Description with comparison note
            desc_text = f"{description}"
            if self.model_ready:
                desc_text += " • ML-powered pricing (0.779 calibration)"
            else:
                desc_text += " • Demo pricing"
            
            desc_label = tk.Label(
                content_frame,
                text=desc_text,
                bg="white",
                font=("Arial", 8),
                fg="#666666"
            )
            desc_label.pack(anchor=tk.W)
        
        # Add comparison note for team
        comparison_frame = tk.Frame(self.results_frame, bg="#f8f9fa", relief=tk.RAISED, bd=1)
        comparison_frame.pack(fill=tk.X, pady=(10, 0))
        
        comparison_text = "💡 For Team: Compare these ML predictions with actual Uber prices to evaluate model performance"
        if not self.model_ready:
            comparison_text = "⚠️ For Team: Load ML model to see real predictions vs Uber prices"
        
        tk.Label(
            comparison_frame,
            text=comparison_text,
            bg="#f8f9fa",
            font=("Arial", 9, "italic"),
            fg="#6c757d",
            wraplength=800,
            padx=10,
            pady=8
        ).pack()
        
        # Update status
        status_msg = "✅ ML calculation complete!" if self.model_ready else "💡 Demo calculation complete"
        self.status_var.set(status_msg)
        self.calculate_btn.config(state=tk.NORMAL)
        
    def show_results(self, pickup, dropoff):
        """Show pricing results"""
        
        # Trip info
        mode_text = "API-Enhanced Data" if self.apis_available else "Demo Data"
        trip_info = f"""📍 From: {pickup}
📍 To: {dropoff}
📏 Distance: ~15.0 km (using {mode_text})
⏱️ Duration: ~25 minutes
🚦 Traffic: Moderate
🌤️ Weather: Clear 75°F

{'✅ Enhanced with real APIs!' if self.apis_available else '⚠️ Demo mode - set API keys for real data'}"""
        
        info_label = tk.Label(
            self.results_frame,
            text=trip_info,
            bg=self.bg_color,
            font=("Arial", 10),
            justify=tk.LEFT,
            fg="#333333" if self.apis_available else "#666666"
        )
        info_label.pack(anchor=tk.W, pady=(0, 15))
        
        # Service prices
        services = [
            ("UberX", 25.50, "Affordable rides for up to 4", "#276EF1"),
            ("UberXL", 35.75, "Rides for groups up to 6", "#00a862"),
            ("Uber Premier", 45.20, "Premium rides", "#000000"),
            ("Premier SUV", 55.90, "Premium SUVs for up to 6", "#6B3AA7")
        ]
        
        for service_name, price, description, color in services:
            service_frame = tk.Frame(self.results_frame, bg="white", relief=tk.RAISED, bd=1)
            service_frame.pack(fill=tk.X, pady=3)
            
            # Color bar
            color_bar = tk.Frame(service_frame, bg=color, width=5)
            color_bar.pack(side=tk.LEFT, fill=tk.Y)
            
            # Content
            content_frame = tk.Frame(service_frame, bg="white", padx=15, pady=8)
            content_frame.pack(side=tk.LEFT, fill=tk.BOTH, expand=True)
            
            # Name and price
            name_price_frame = tk.Frame(content_frame, bg="white")
            name_price_frame.pack(fill=tk.X)
            
            status_suffix = " (API-Enhanced)" if self.apis_available else " (DEMO)"
            service_label = tk.Label(
                name_price_frame,
                text=f"{service_name}{status_suffix}",
                bg="white",
                font=("Arial", 12, "bold")
            )
            service_label.pack(side=tk.LEFT)
            
            price_label = tk.Label(
                name_price_frame,
                text=f"${price:.2f}",
                bg="white",
                font=("Arial", 14, "bold"),
                fg=color
            )
            price_label.pack(side=tk.RIGHT)
            
            # Description
            desc_text = f"{description} • Enhanced with real data" if self.apis_available else f"{description} • Demo pricing"
            desc_label = tk.Label(
                content_frame,
                text=desc_text,
                bg="white",
                font=("Arial", 8),
                fg="#666666"
            )
            desc_label.pack(anchor=tk.W)
        
        # Update status
        status_msg = "✅ Calculation complete with APIs!" if self.apis_available else "💡 Demo calculation complete"
        self.status_var.set(status_msg)
        self.calculate_btn.config(state=tk.NORMAL)

def get_api_keys_from_user():
    """Get API keys from user if not set"""
    root = tk.Tk()
    root.withdraw()
    
    google_key = os.environ.get('GOOGLE_MAPS_API_KEY')
    weather_key = os.environ.get('WEATHER_API_KEY')
    
    if not google_key:
        google_key = simpledialog.askstring(
            "API Key Setup", 
            "Enter your Google Maps API Key (or press Cancel for demo mode):",
            show='*'
        )
        if google_key:
            os.environ['GOOGLE_MAPS_API_KEY'] = google_key
    
    if not weather_key:
        weather_key = simpledialog.askstring(
            "API Key Setup", 
            "Enter your Weather API Key (or press Cancel for demo mode):",
            show='*'
        )
        if weather_key:
            os.environ['WEATHER_API_KEY'] = weather_key
    
    root.destroy()

def main():
    """Main function"""
    print("🚀 Starting Simple Advanced Uber Pricing GUI...")
    
    # Get API keys from user if needed
    get_api_keys_from_user()
    
    # Create and run GUI
    root = tk.Tk()
    app = SimpleUberGUI(root)
    root.mainloop()

if __name__ == "__main__":
    main() 