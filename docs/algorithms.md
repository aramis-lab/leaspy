# Algorithms

## Fit
### Prerequisites
   - Data format and preprocessing steps.
### Running Task
   - How to run the fit algorithm.
   - Example commands and code snippets.
### Output
   - What results to expect from the fitting process.
   - DataFrame object 
   - Plotting object 
### Setting Options (Different Models)
   - How to set specific options for different types of models.
   - Customizing parameters for logistic, joint, mixture, and covariate models.
## Personalize
## Estimate
## Simulate

This section describes the procedure for simulating new patient data based on a fitted Leaspy model and user-defined parameters. The simulation method involves the following steps:

**Step 1: Generation of Individual Parameters**
For each simulated patient, individual parameters (tau, xi, and sources) are sampled from normal distributions defined by the model’s mean and standard deviation. These model parameters come from a previously fitted Leaspy model, provided by the user. 

**Step 2: Generation of Visit Times**
Visit times are generated based on user-specified visit parameters, such as the number of visits, spacing between visits, and follow-up duration. These parameters are provided through a dictionary.

**Step 3: Estimation of Observations**
The estimate function from Leaspy is used to compute the patient observations at the generated visit times, based on the individual parameters. To reflect variability in the observations, beta-distributed noise is added, appropriate for modeling outcomes in a logistic framework.

### Prerequisites
To run a simulation, the following variables are required:
- A fitted Leaspy model (see the `fit` function), used for both parameter sampling (step 1) and the estimate function (step 3).
- A dictionary of visit parameters, specifying the number, type, and timing of visits (used in step 2).
- An `AlgorithmSettings` object, configured for simulation and including:
  - The name of the features to simulate.
  - The visit parameter dictionary.

### Running the Task

```python
>>> from leaspy.algo import AlgorithmSettings
>>> visits_params = {
        'patient_nb': 200,
        'visit_type': "regular",
        'regular_visit': 1,
        'first_visit_mean': 0.,
        'first_visit_std': 0.4,
        'time_follow_up_mean': 11,
        'time_follow_up_std': 0.5,
        'distance_visit_mean': 2 / 12,
        'distance_visit_std': 0.75 / 12,
    }
>>> simulation_settings = AlgorithmSettings(
        "simulation",
        seed=0,
        features=["MDS1_total", "MDS2_total", "MDS3_off_total"],
        visit_parameters=visits_params
    )
>>> simulated_data = leaspy_logistic.simulate(simulation_settings)
Simulate with `simulation` took: 3s
>>> print(simulated_data.data.to_dataframe().set_index(['ID', 'TIME']).head())
            MDS1_total   MDS2_total  MDS3_total   
ID   TIME
0  48.092     0.113991	   0.344018	   0.228870 
   49.092     0.177138	   0.291440	   0.332267  
   50.092     0.256084	   0.253093	   0.314241	 
   51.092     0.174059	   0.344921	   0.313776 
   52.092     0.245289	   0.400363	   0.310065 
```

### Output

The output is a Data object with ID, TIME and simulated values of each feature. 

### Setting options

There are three options to simulate the visit times in Leaspy, which can be specified in visit_param dictionary: 
- `random`: Visit times and intervals are sampled from normal distributions.
- `regular`: Visits occur at regular intervals, defined by regular_visit. 
- `dataframe`: Custom visit times are provided directly via a DataFrame.

Refer to the docstring for further details.


## Data Generalities
- monotonicity
- NaN 
- number of visits 
- outliers
- not enough patients 
- parameters don't converge 
- score don't progress