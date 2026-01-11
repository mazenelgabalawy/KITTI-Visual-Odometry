# Setting up the environment
Create a python venv and install all dependencies using
`pip install -r requirments.txt`

# Running Base Pipeline
To run the application, use the following command:
```python VO_base.py ```
The program processes all 11 sequences. To change the required sequence, modify the range in the for loop in the main function of `VO_base.py`.
To change input and output folders, modify the respective variables in the main function of `VO_base.py`.

# Running Constrained Pipeline
To run the application, use the following command:
```python VO_constrained.py ```
The program processes all 11 sequences. To change the required sequence, modify the range in the for loop in the main function of `VO_constrained.py`.
To change input and output folders, modify the respective variables in the main function of `VO_constrained.py`.


# Creating EVO results
To generate EVO results, run the following command:

```bash ./create_results.sh <path_to_output_folder>```

Replace `<path_to_output_folder>` with the path to the folder containing the output data,e.g:

```bash ./create_results.sh ./output_constrained/```

By default, the script will process all sequences in the specified folder. If you want to process a specific sequence, you can provide the sequence name as an additional argument:

```bash ./create_results.sh <path_to_output_folder> <sequence_name>```

Make sure to replace `<sequence_name>` with the actual name of the sequence you want to process, e.g:

```bash ./create_results.sh ./output_constrained/ 01```

# Viewing the results
After running the script, you can find the generated EVO results in the specified output folder. The results include text files and plots for APE, RPE, and trajectory comparisons.

![APE Map](./output_constrained/00/ape_plot_map.png)
![Trajectory Plot](./output_constrained/00/trajectory_plot_xyz.png)
