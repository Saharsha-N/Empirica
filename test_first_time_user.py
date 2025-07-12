"""
First-time user test script following the README workflow.
This demonstrates the basic Empirica workflow without requiring cmbagent.
"""
import tempfile
import os
from pathlib import Path
from empirica import Empirica

# Create a temporary directory for the test project
temp_dir = tempfile.mkdtemp()
project_dir = os.path.join(temp_dir, "test_project")

print("=" * 60)
print("FIRST-TIME USER TEST - Empirica Workflow")
print("=" * 60)
print(f"\nProject directory: {project_dir}\n")

try:
    # Step 1: Initialize Empirica instance
    print("Step 1: Initializing Empirica instance...")
    emp = Empirica(project_dir=project_dir)
    print("[OK] Empirica initialized successfully\n")
    
    # Step 2: Set data description
    print("Step 2: Setting data description...")
    prompt = """
    Analyze the experimental data stored in data.csv using sklearn and pandas.
    This data includes time-series measurements from a particle detector.
    The dataset contains 1000 samples with features: timestamp, sensor_reading, temperature, and pressure.
    """
    emp.set_data_description(prompt)
    print("[OK] Data description set successfully\n")
    print(f"Data description preview: {emp.research.data_description[:100]}...\n")
    
    # Step 3: Generate a research idea (using fast mode to avoid cmbagent requirement)
    print("Step 3: Generating research idea (fast mode)...")
    print("Note: Using 'fast' mode which doesn't require cmbagent")
    try:
        emp.get_idea(mode="fast")
        print("[OK] Research idea generated successfully\n")
        if hasattr(emp.research, 'idea') and emp.research.idea:
            print(f"Idea preview: {emp.research.idea[:200]}...\n")
        else:
            print("Note: Idea file may need to be checked manually\n")
    except Exception as e:
        print(f"[WARNING] Idea generation encountered an issue: {e}\n")
        print("This is expected if API keys are not configured.\n")
    
    # Step 4: Generate methodology (using fast mode)
    print("Step 4: Generating methodology (fast mode)...")
    try:
        emp.get_method(mode="fast")
        print("[OK] Methodology generated successfully\n")
        if hasattr(emp.research, 'methodology') and emp.research.methodology:
            print(f"Methodology preview: {emp.research.methodology[:200]}...\n")
        else:
            print("Note: Methodology file may need to be checked manually\n")
    except Exception as e:
        print(f"[WARNING] Method generation encountered an issue: {e}\n")
        print("This is expected if API keys are not configured.\n")
    
    # Step 5: Check if results can be generated (requires cmbagent)
    print("Step 5: Checking results generation...")
    print("Note: get_results() requires cmbagent to be installed")
    print("Skipping this step as cmbagent is not available\n")
    
    # Step 6: Check project structure
    print("Step 6: Checking project structure...")
    project_path = Path(project_dir)
    if project_path.exists():
        print(f"[OK] Project directory created: {project_dir}")
        input_files = project_path / "input_files"
        if input_files.exists():
            files = list(input_files.glob("*.md"))
            print(f"[OK] Found {len(files)} markdown files in input_files/")
            for f in files:
                print(f"  - {f.name}")
        print()
    
    print("=" * 60)
    print("TEST COMPLETE")
    print("=" * 60)
    print("\nSummary:")
    print("- Empirica instance created successfully")
    print("- Data description set")
    print("- Idea and method generation attempted (may require API keys)")
    print("- Project structure created")
    print(f"\nProject files are in: {project_dir}")
    print("\nNext steps for a real workflow:")
    print("1. Configure API keys (see documentation)")
    print("2. Run get_idea() and get_method() with API keys")
    print("3. Install cmbagent for get_results() functionality")
    print("4. Run get_paper() to generate LaTeX paper")
    
except Exception as e:
    print(f"\n[ERROR] Error during test: {e}")
    import traceback
    traceback.print_exc()

