import numpy as np
import pandas as pd

class SimpleChargingFacility:
    def __init__(self, total_spaces, penalty_rate, lambda_val, sim_time):
        self.total_spaces = total_spaces
        self.penalty_rate = penalty_rate  # $/minute for overstaying
        self.lambda_val = lambda_val      # arrival rate (vehicles/hour)
        self.sim_time = sim_time          # simulation duration (hours)
        self.occupied = []                # tracks (arrival, departure) times
        self.charging_demands = []        # kWh
        self.overstay_times = []          # hours
        self.needed_charge_times = []     # hours

    def run(self):
        # Generate arrival times (Poisson process)
        arrival_times = np.cumsum(np.random.exponential(1/self.lambda_val, 
                              int(2*self.lambda_val*self.sim_time)))
        arrival_times = arrival_times[arrival_times < self.sim_time]
        
        # Generate parameters for each vehicle
        self.charging_demands = np.random.uniform(20, 80, len(arrival_times))  # kWh
        charge_rate = 50  # kW (updated to 50kW as requested)
        self.needed_charge_times = self.charging_demands / charge_rate  # hours
        self.overstay_times = np.random.exponential(0.5, len(arrival_times))  # hours
        
        # Process each vehicle
        self.occupied = []
        for a_time, needed_time, xi_j in zip(arrival_times, 
                                           self.needed_charge_times,
                                           self.overstay_times):
            actual_time = max(needed_time, xi_j)
            self.occupied.append((a_time, a_time + actual_time))

def generate_dataframes(facility):
    # 1. Time-Series Utilization (minute resolution)
    time_points = np.linspace(0, facility.sim_time, 1440)
    utilization = []
    for t in time_points:
        active = sum(start <= t < end for start, end in facility.occupied)
        utilization.append(active / facility.total_spaces * 100)
    
    time_series_df = pd.DataFrame({
        'Time (hours)': time_points,
        'Utilization (%)': utilization
    })

    # 2. Detailed Event Log
    event_data = []
    for (start, end), x_j, needed_time, xi_j in zip(facility.occupied,
                                                   facility.charging_demands,
                                                   facility.needed_charge_times,
                                                   facility.overstay_times):
        overstay_minutes = max(0, (xi_j - needed_time) * 60)  # Convert to minutes
        penalty = overstay_minutes * facility.penalty_rate
        
        event_data.append({
            'Arrival Time': start,
            'Departure Time': end,
            'Charging Demand (kWh)': x_j,
            'Required Charge Time (hr)': needed_time,
            'Parking Duration (hr)': xi_j,
            'Overstayed (min)': overstay_minutes,
            'Penalty ($)': penalty
        })
    
    event_df = pd.DataFrame(event_data)

    # 3. Summary Statistics (with NaN protection)
    overstays = event_df[event_df['Overstayed (min)'] > 0]
    avg_overstay = overstays['Overstayed (min)'].mean() if len(overstays) > 0 else 0
    
    summary_df = pd.DataFrame({
        'Metric': [
            'Total Spaces', 
            'Vehicles Processed',
            'Peak Utilization (%)',
            'Avg Utilization (%)',
            'Vehicles Overstayed',
            'Overstay Rate (%)',
            'Avg Overstay Time (min)',
            'Max Overstay Time (min)',
            'Total Penalty Revenue ($)',
            'Avg Penalty ($) per Overstay'
        ],
        'Value': [
            facility.total_spaces,
            len(event_df),
            max(utilization),
            np.mean(utilization),
            len(overstays),
            100 * len(overstays) / len(event_df) if len(event_df) > 0 else 0,
            avg_overstay,
            overstays['Overstayed (min)'].max() if len(overstays) > 0 else 0,
            event_df['Penalty ($)'].sum(),
            overstays['Penalty ($)'].mean() if len(overstays) > 0 else 0
        ]
    })
    
    return time_series_df, event_df, summary_df

# Example Usage with your requested parameters
if __name__ == "__main__":
    # Initialize facility with your specified parameters
    facility = SimpleChargingFacility(
        total_spaces=29,
        penalty_rate=1,    # $1 per minute overstayed
        lambda_val=20,     # 20 arrivals per hour
        sim_time=12        # 12-hour simulation
    )
    
    # Run simulation
    facility.run()
    
    # Generate reports
    time_series_df, event_df, summary_df = generate_dataframes(facility)

    print("=== Summary Statistics ===")
    print(summary_df.to_string(index=False))
    
    print("\n=== First 5 Events ===")
    print(event_df.head().to_string(index=False))
    
    print("\n=== Utilization Over Time ===")
    print(f"Tracked at 1-minute intervals ({len(time_series_df)} data points)")