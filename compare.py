import cocopp

# List all the result folders you want to compare
algorithm_result_folders = [
    "exdata/exdata/CMAES", # Replace with actual folder names from your runs
    "exdata/exdata/GWO",
    "exdata/exdata/LBFGSB",
    "exdata/exdata/Spiral-LSHADE",
    "exdata/exdata/SciPyDE",
    "exdata/exdata/PSO",
    "exdata/exdata/SSA",    
    "exdata/exdata/GA",    
    "exdata/exdata/LSHADE",    
    "exdata/exdata/NLSHADE_RSP",    
    "exdata/exdata/JADE",    
    "exdata/exdata/jSO",    

    # Add more folders for other algorithms
]

# Run cocopp to generate the comparison report
cocopp.main(algorithm_result_folders)