import numpy as np
import pandas as pd
import tensorflow as tf
from tensorflow.keras import layers, models
import matplotlib.pyplot as plt
import seaborn as sns
import random
import time
import tkinter as tk
from tkinter import ttk
from flask import Flask, render_template, jsonify

# Simulate smart grid power consumption dataset
num_houses = 5
num_appliances = 6
num_days = 30

def generate_smart_grid_data():
    appliances = ['Fridge', 'AC', 'TV', 'Lights', 'Washing Machine', 'Microwave']
    data = []
    
    for house in range(1, num_houses+1):
        for day in range(1, num_days+1):
            total_power = 0
            appliance_data = {}
            voltage = random.uniform(220, 240)
            current = random.uniform(5, 15)
            power_factor = random.uniform(0.8, 1.0)
            energy_theft = random.choice([0, random.uniform(2, 8)])  # Increased theft difference
            
            for app in appliances:
                expected_usage = random.uniform(1, 4) * (random.randint(5, 12))  # Slightly larger range
                actual_usage = expected_usage + random.uniform(-1, 1)  # Greater expected-actual variation
                appliance_data[f'Expected_{app}'] = expected_usage
                appliance_data[f'Actual_{app}'] = actual_usage
                total_power += actual_usage
            
            data.append({
                'House': house,
                'Day': day,
                'Voltage': voltage,
                'Current': current,
                'Power Factor': power_factor,
                'Total Power': total_power + energy_theft,
                'Energy Theft': energy_theft,
                **appliance_data
            })
    
    return pd.DataFrame(data)

# Generate and save dataset
dataset = generate_smart_grid_data()
dataset.to_csv("smart_grid_power_usage.csv", index=False)
print("Smart grid dataset generated and saved.")

# Load dataset
data = pd.read_csv("smart_grid_power_usage.csv")

# Autoencoder for anomaly detection
def build_autoencoder():
    model = models.Sequential([
        layers.Dense(16, activation='relu', input_shape=(4,)),
        layers.Dense(8, activation='relu'),
        layers.Dense(16, activation='relu'),
        layers.Dense(4, activation='sigmoid')
    ])
    model.compile(optimizer='adam', loss='mse')
    return model

autoencoder = build_autoencoder()
autoencoder.fit(data[['Total Power', 'Voltage', 'Current', 'Power Factor']],
                data[['Total Power', 'Voltage', 'Current', 'Power Factor']], epochs=50)

# GUI for visualization
def display_house_graph(house):
    plt.figure(figsize=(12, 6))
    expected = [f'Expected_{app}' for app in ['Fridge', 'AC', 'TV', 'Lights', 'Washing Machine', 'Microwave']]
    actual = [f'Actual_{app}' for app in ['Fridge', 'AC', 'TV', 'Lights', 'Washing Machine', 'Microwave']]
    house_data = data[data['House'] == house].mean()
    
    x_labels = ['Fridge', 'AC', 'TV', 'Lights', 'Washing Machine', 'Microwave']
    expected_vals = [house_data[col] for col in expected]
    actual_vals = [house_data[col] for col in actual]
    
    x = np.arange(len(x_labels))
    width = 0.35
    
    fig, ax = plt.subplots()
    ax.bar(x - width/2, expected_vals, width, label='Expected')
    ax.bar(x + width/2, actual_vals, width, label='Actual')
    
    ax.set_xlabel("Appliance")
    ax.set_ylabel("Power Consumption (kWh)")
    ax.set_title(f"House {house} Power Consumption")
    ax.set_xticks(x)
    ax.set_xticklabels(x_labels)
    ax.legend()
    plt.show()

def display_street_graph():
    plt.figure(figsize=(12, 6))
    street_data = data.groupby('Day').sum()
    plt.plot(street_data.index, street_data['Total Power'], label='Total Power Consumption', marker='o')
    plt.xlabel("Day")
    plt.ylabel("Power Consumption (kWh)")
    plt.title("Street-Level Power Consumption")
    plt.legend()
    plt.grid()
    plt.show()

def display_energy_theft_graph():
    plt.figure(figsize=(12, 6))
    theft_data = data.groupby('Day').sum()['Energy Theft']
    plt.fill_between(theft_data.index, theft_data, color='red', alpha=0.5)
    plt.xlabel("Day")
    plt.ylabel("Stolen Energy (kWh)")
    plt.title("Energy Theft Over Time")
    plt.show()

root = tk.Tk()
root.title("Smart Grid Dashboard")
root.geometry("600x500")

tk.Label(root, text="Select Visualization:", font=("Arial", 12)).pack()
options = {"House Power Consumption": display_house_graph,
           "Street-Level Consumption": display_street_graph,
           "Energy Theft Trends": display_energy_theft_graph}

graph_selection = ttk.Combobox(root, values=list(options.keys()))
graph_selection.pack()

tk.Button(root, text="Display Graph", font=("Arial", 12), command=lambda: options[graph_selection.get()]() if graph_selection.get() else None).pack()

tk.Label(root, text="Select House for Power Graph:", font=("Arial", 12)).pack()
house_selection = ttk.Combobox(root, values=[1, 2, 3, 4, 5])
house_selection.pack()

tk.Button(root, text="Display House Graph", font=("Arial", 12), command=lambda: display_house_graph(int(house_selection.get())) if house_selection.get() else None).pack()

root.mainloop()
