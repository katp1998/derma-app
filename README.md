Clinical Attributes: (take values 0, 1, 2, 3, unless otherwise indicated) 
1 Erythema 
    2 Scaling 
    3 definite borders 
    4 Itching 
    5 koebner phenomenon 
    6 polygonal papules 
    7 follicular papules 
    8 oral mucosal involvement 
    9 knee and elbow involvement 
    10 scalp involvement 
    11 family history (0 or 1) 
    34 Age 
Histopathological Attributes: (take values 0, 1, 2, 3) 
    12 melanin incontinence 
    13 eosinophils in the infiltrate 
    14 PNL infiltrate 
    15 fibrosis of the papillary dermis 
    16 Exocytosis 
    17 Acanthosis 
    18 Hyperkeratosis 
    19 Parakeratosis 
    20 clubbing of the rete ridges 
    21 elongation of the rete ridges 
    22 thinning of the suprapapillary epidermis 
    23 spongiform pustule 
    24 munro microabcess 
    25 focal hypergranulosis 
    26 disappearance of the granular layer 
    27 vacuolisation and damage of basal layer 
    28 Spongiosis 
    29 saw-tooth appearance of retes 
    30 follicular horn plug
    31 perifollicular parakeratosis 
    32 inflammatory monoluclear inflitrate 
    33 band-like infiltrate 
 
This data set contains 34 attributes. 35th is the class label, i.e., the disease name. The names and id numbers of the patients were removed from the database. 
 
The differential diagnosis of erythemato-squamous diseases is a real problem in dermatology. They all share the clinical features of erythema and scaling, with very little differences. The diseases in this group are psoriasis, seboreic dermatitis, lichen planus, pityriasis rosea, cronic dermatitis, and pityriasis rubra pilaris. Usually, a biopsy is necessary for the diagnosis but unfortunately, these diseases share many histopathological features as well. Another difficulty for the differential diagnosis is that a disease may show the features of another disease at the beginning stage and may have the characteristic features at the following stages. Patients were first evaluated clinically with 12 features. Afterwards, skin samples were taken for the evaluation of 22 histopathological features. The values of the histopathological features are determined by an analysis of the samples under a microscope. 
 
In the dataset constructed for this domain, the family history feature has the value 1 if any of these diseases has been observed in the family, and 0 otherwise. The age feature simply represents the age of the patient. Every other feature (clinical and histopathological) was given a degree in the range of 0 to 3. Here, 0 indicates that the feature was not present, 3 indicates the largest amount possible, and 1, 2 indicate the relative intermediate values. 
 
Assuming that the list of diseases in this group are complete (total 6 types), use the dataset to address the following list of objectives: 

1.	Determine the type of disease based on the patient’s Age. Use gradient descent (GD) to build your regression model (model 1). Start by writing the GD algorithm and then implement it. [20 Marks] 

2.	Use random forest on the clinical as well as histopathological attributes to classify the disease type (model2). [15 Marks] 

3.	Use kNN on the clinical attributes and histopathological attributes to classify the disease type and report your accuracy (model3). [15 Marks] 

4.	Finally, use two different clustering algorithms and see how well these attributes can determine the disease type (model4 and model5). [20 Marks] 

Make sure to report your actual model for each of the above. This means providing appropriate details containing the features used, parameters learned/estimated, and any inputs (e.g., number of clusters, error limit) that go into the technique. When appropriate, perform multiple runs of the same technique and report average values or their spread. 

Compare and contrast the models you built. Having done both classification and clustering on the same dataset, what can you say about this data and/or the techniques you used? [10 Marks] 

You must hand in both the report and code. However, the marks will come from what you provide in the report. You must demonstrate how you went about each point above and provide any outputs in the report. Any detail left out of the report but is in the code will not be awarded marks.  You must write the report for a non-technical manager to understand your findings, code and understand how the conclusions you have come to are robust. 
 
Additional marks will be given for:

•	APA referencing [5 Marks]
•	Creativity, clarity, above and beyond class materials, work is comparable to real-world solutions, and complexity. [15 Marks]