# IEDA_FYP
HKSTP EV Parking data 2024-2025

Data visualization on the usage by fraction on daily basis

simulaiton of overstaying situation using arrival rate and charging rate(50kWh ) of the HKSTP, using automatica's utlilities function . 
Input:     facility = SimpleChargingFacility(
        total_spaces=29,
        penalty_rate=4,    # $1 per minute overstayed
        lambda_val=10,     # 20 arrivals per hour
        sim_time=12        # 12-hour simulation
    )
Example output:

=== Summary Statistics ===
                      Metric       Value
                Total Spaces   29.000000
          Vehicles Processed  122.000000
        Peak Utilization (%)   65.517241
         Avg Utilization (%)   37.595785
         Vehicles Overstayed   24.000000
           Overstay Rate (%)   19.672131
     Avg Overstay Time (min)   30.116364
     Max Overstay Time (min)  114.464625
   Total Penalty Revenue ($) 2891.170920
Avg Penalty ($) per Overstay  120.465455
