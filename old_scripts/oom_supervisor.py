import subprocess
import time

# Path to the target Python program
target_program = "target.py"

def run_target():
    try:
        # Run the target program and capture output and errors
        result = subprocess.run(
            ["python", target_program],
            stdout=subprocess.PIPE,
            stderr=subprocess.PIPE,
            text=True
        )

        # Output for debugging
        print("Return Code:", result.returncode)
        print("Stdout:\n", result.stdout)
        print("Stderr:\n", result.stderr)

        # Check if there's an out-of-memory error
        if ("MemoryError" in result.stderr or 
            "std::bad_alloc" in result.stderr or 
            "Out of memory" in result.stderr or 
            result.returncode != 0 and "Killed" in result.stderr):
            print("Detected OOM. Restarting...")
            return False  # Indicate failure due to memory
        else:
            print("Program completed without OOM.")
            return True

    except Exception as e:
        print(f"Unexpected error: {e}")
        return False

# Main loop
while True:
    success = run_target()
    if success:
        break
    print("Retrying in 2 seconds...")
    time.sleep(2)
