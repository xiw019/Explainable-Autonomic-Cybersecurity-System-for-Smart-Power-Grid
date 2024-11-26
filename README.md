# Explainable Autonomic Cybersecurity System for Smart Power Grid

This repository contains the development and implementation of an Explainable Autonomic Cybersecurity System tailored for the Smart Power Grid. 

The system leverages machine learning and explainable AI (XAI) to detect, mitigate, and analyze cybersecurity threats in critical infrastructure systems.

## Project Objectives
**Explainability:** 
Integrate XAI techniques to enhance trust and interpretability for domain experts, operators, and stakeholders.


**Scalability:** Ensure the framework scales effectively for diverse grid infrastructures by providing a customizable test-bed. 

## Testbed Setup & Usage 
The `DT_explainer` module includes functionalities to:
1. Explain predictions made by a decision tree model.
2. Visualize the structure of the decision tree for better interpretability.
3. Identify and rank features based on their importance in the model's decision-making process.
### 1. Import the Module
Start by importing the `DT_explainer` module into your Python script or Jupyter notebook.
```python
from DT_explainer import DTExplainer
```

### 2. Create and Initialize the  Explainer 
```python
explainer = DTExplainer(trained_decision_tree_model)
```
 
explanations = explainer.explain(data_to_explain)



## Explainer Setup & Usage

## Auto-mitigation Setup