#!/bin/bash

# Configuration
DEVICE="0000:88:00.1"  # Your FPGA device ID
SAMPLE_INTERVAL=1      # Sampling interval in seconds
LOG_FILE="power_log.csv"

# Create CSV header
echo "Timestamp,Power(W)" > $LOG_FILE

# Function to get current power consumption
get_power() {
    # Extract power value in Watts, accounting for the specific format shown
    power_val=$(xbutil examine --report electrical --device $DEVICE | grep -i "Power[ ]*:" | grep -v "Max\|Warning" | awk -F: '{print $2}' | awk '{print $1}' | tr -d '\r\n')
    echo $power_val
}

# Start power monitoring in background
monitor_power() {
    while [ -f ".monitoring" ]; do
        POWER=$(get_power)
        TIMESTAMP=$(date +"%Y-%m-%d %H:%M:%S")
        
        # Only log if we got a valid power value
        if [[ ! -z "$POWER" && "$POWER" =~ ^[0-9]+(\.[0-9]+)?$ ]]; then
            echo "$TIMESTAMP,$POWER" >> $LOG_FILE
        else
            echo "Warning: Invalid power reading obtained, skipping this sample"
            # Debug output to help diagnose issues
            echo "Debug: xbutil output snippet:"
            xbutil examine --report electrical --device $DEVICE | grep -i "Power" | head -3
        fi
        
        sleep $SAMPLE_INTERVAL
    done
}

# Start monitoring flag
touch .monitoring

# Start power monitoring in background
monitor_power &
MONITOR_PID=$!

echo "Power monitoring started. Log file: $LOG_FILE"
echo "Running bitstream..."

# Record start time
START_TIME=$(date +%s)

# Run the bitstream
# Uncomment the line below and adjust as needed for your environment
make run TARGET=hw PLATFORM=$XDEVICE

# Record end time
END_TIME=$(date +%s)

# Stop monitoring
rm .monitoring
wait $MONITOR_PID

# Calculate execution time
EXEC_TIME=$((END_TIME - START_TIME))
echo "Execution time: $EXEC_TIME seconds"

# Process the data - make sure each line has the format "timestamp,power_value"
# Create a temporary clean file
TMP_LOG_FILE="${LOG_FILE}.tmp"
grep -v "^$" $LOG_FILE | grep "," > $TMP_LOG_FILE

# Calculate energy consumption
echo "Calculating energy consumption..."

# These calculations assume proper CSV format with "Timestamp,Power" on each line
AVG_POWER=$(awk -F, 'NR>1 {if ($2 ~ /^[0-9]+(\.[0-9]+)?$/) {sum+=$2; count+=1}} END {if (count > 0) print sum/count; else print 0}' $TMP_LOG_FILE)
TOTAL_ENERGY=$(awk -F, 'NR>1 {if ($2 ~ /^[0-9]+(\.[0-9]+)?$/) {sum+=$2}} END {print sum*'$SAMPLE_INTERVAL'}' $TMP_LOG_FILE)
WATT_HOURS=$(echo "$TOTAL_ENERGY/3600" | bc -l)

echo "Results:"
echo "  Average Power: $AVG_POWER Watts"
echo "  Total Energy: $TOTAL_ENERGY Joules (Watt-seconds)"
echo "  Total Energy: $WATT_HOURS Watt-hours"
echo "Full power log saved to $LOG_FILE"

# Optional: Create a visualization of the power consumption over time
# Uncomment the section below if you have gnuplot installed
# echo "Creating power consumption graph..."
# gnuplot <<EOF
# set terminal png size 800,600
# set output "power_consumption.png"
# set title "FPGA Power Consumption Over Time"
# set xlabel "Time"
# set ylabel "Power (W)"
# set xdata time
# set timefmt "%Y-%m-%d %H:%M:%S"
# set format x "%H:%M:%S"
# set grid
# plot "$TMP_LOG_FILE" using 1:2 with lines title "Power Consumption"
# EOF
# echo "Power consumption graph saved as power_consumption.png"

# Clean up temporary file
rm -f $TMP_LOG_FILE