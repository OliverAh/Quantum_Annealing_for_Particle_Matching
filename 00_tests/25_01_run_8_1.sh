#!/bin/bash

# Define the functions

function func1() {
    # Execute the first script file
    python 25_01_run_8_1.py >> 01_out/sub_8_1/out.txt 2>&1
}

function func2() {
    # Execute the second script file
    git add -A
    git commit -m"state after a subrun of sub8_1"
    git push
}

function func3() {
    # Execute the third script file
    python 25_01_run_8_1_reset.py >> 01_out/sub_8_1/out.txt 2>&1
}


function func4() {
    # Directory to monitor
    DIRECTORY="01_out/sub_8_1/samples"  # <-- Update this path

    # Count the number of files in the directory (including subdirectories)
    FILE_COUNT=$(find "$DIRECTORY" -type f | wc -l)

    # Display the number of files
    echo "Number of files in $DIRECTORY: $FILE_COUNT"

    # Check if the file count equals 300
    if [ "$FILE_COUNT" -eq 352 ]; then
        echo "File count has reached 352. Exiting the loop."
        return 1  # Non-zero return value to signal loop exit
    else
        return 0  # Zero return value to continue the loop
    fi
}

# Loop and call the functions
for (( i=1; i<=100; i++ ))
do
    echo "Iteration $i:"
    func1
    func2
    func3
    
    func4

    # Check the exit status of func4 to determine if the loop should exit
    if [ $? -ne 0 ]; then
        # Break out of the loop if func4 signals to exit
        break
    fi

    echo "---------------------"
done