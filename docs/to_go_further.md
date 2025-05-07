# To Go Further

## Model Implementation
### DAG Intuition and Structure for Models

## What kind of scientific question could be answered with leaspy? 

Different papers have been published trying to answer clinical question using the software.

### Used in different context

__Different chronic diseases:__ The model has been used to describe very different chronic diseases as Hungtington {cite}`koval_forecasting_2022`, Alzheimer {cite}`maheux_forecasting_2023`, Cerebral Autosomal Dominant Arteriopathy with Subcortical Infarcts Leukoencephalopathy (CADASIL) {cite}`kaisaridi_determining_2025`, Amyothrophic Lateral Sclerosis {cite}`ortholand_interaction_2023`, Ataxy {cite}`moulaire_temporal_2023`, Parkinson [REF].
ALS
__Many types of data:__ Different types of data have been analysed from clinical score to biomarkers such as clinical score and brain markers {cite}`koval_ad_2021` and events [Juliette]. For longitudinal data progression where used from linear [REF?] and logistic [REF] to ordinal [Paul].  Note that the model has been already used in XX clinical outcomes [Sofia] and has been shown quite robust to missing data [REF Couronne].

### Used for different tasks

__Describe the joint progression of multiple outcomes__ 

__Describe disease heterogeneity:__ Post-hoc analysis of the individual variability to describe disease heterogeneity were conducted using a supervised approach for for Amyotrophic Lateral Sclerosis [REF Juliette] and Ataxy [REF Paul Emilien ?] as well as an unsupervised approaches for CADASIL [REF Sofia].

__Improve clinical trials:__ The model has been shown useful to select patients for clinical trials in order to increase the sensibility of the trial [Paper Etienne]. [Paper Maylis & PE]

__Make predictions:__ TADPOLE and others 

### References

- Couronne, Raphael, Marie Vidailhet, Jean Christophe Corvol, Stephane Lehericy, et Stanley Durrleman. « Learning Disease Progression Models With Longitudinal Data and Missing Values ». In 2019 IEEE 16th International Symposium on Biomedical Imaging (ISBI 2019), 1033‑37, 2019. https://doi.org/10.1109/ISBI.2019.8759198.
- Di Folco, Cécile, Raphaël Couronné, Isabelle Arnulf, Graziella Mangone, Smaranda Leu-Semenescu, Pauline Dodet, Marie Vidailhet, Jean-Christophe Corvol, Stéphane Lehéricy, et Stanley Durrleman. « Charting Disease Trajectories from Isolated REM Sleep Behavior Disorder to Parkinson’s Disease ». Movement Disorders n/a, no n/a (2023). https://doi.org/10.1002/mds.29662.
- Kaisaridi, Sofia, Dominique Herve, Aude Jabouley, Sonia Reyes, Carla Machado, Stéphanie Guey, Abbas Taleb, Fanny Fernandes, Hugues Chabriat, et Sophie Tezenas Du Montcel. « Determining Clinical Disease Progression in Symptomatic Patients With CADASIL ». Neurology 104, no 1 (14 janvier 2025): e210193. https://doi.org/10.1212/WNL.0000000000210193.
- Maheux, Etienne, Igor Koval, Juliette Ortholand, Colin Birkenbihl, Damiano Archetti, Vincent Bouteloup, Stéphane Epelbaum, Carole Dufouil, Martin Hofmann-Apitius, et Stanley Durrleman. « Forecasting Individual Progression Trajectories in Alzheimer’s Disease ». Nature Communications 14, no 1 (2023): 761. https://doi.org/10.1038/s41467-022-35712-5.
- Moulaire, Paul, Pierre Emmanuel Poulet, Emilien Petit, Thomas Klockgether, Alexandra Durr, Tetsuo Ashisawa, Sophie Tezenas Du Montcel, et for the READISCA Consortium. « Temporal Dynamics of the Scale for the Assessment and Rating of Ataxia in Spinocerebellar Ataxias ». Movement Disorders 38, no 1 (2023): 35‑44. https://doi.org/10.1002/mds.29255.
- Ortholand, Juliette, Pierre-François Pradat, Sophie Tezenas Du Montcel, et Stanley Durrleman. « Interaction of Sex and Onset Site on the Disease Trajectory of Amyotrophic Lateral Sclerosis ». Journal of Neurology 270, no 12 (2023): 5903‑12. https://doi.org/10.1007/s00415-023-11932-7.


[TODO: literature review Igor, Cecile, Raphael, Jean-Baptiste, Etienne => maybe subselect the most appropriate the fancier]