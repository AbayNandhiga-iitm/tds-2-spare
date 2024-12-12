Dataset Name: media

Basic Description:
       language  type      overall      quality  repeatability
count       0.0   0.0  2652.000000  2652.000000    2652.000000
mean        NaN   NaN     3.047511     3.209276       1.494721
std         NaN   NaN     0.762180     0.796743       0.598289
min         NaN   NaN     1.000000     1.000000       1.000000
25%         NaN   NaN     3.000000     3.000000       1.000000
50%         NaN   NaN     3.000000     3.000000       1.000000
75%         NaN   NaN     3.000000     4.000000       2.000000
max         NaN   NaN     5.000000     5.000000       3.000000

Missing Values:
language         2652
type             2652
overall             0
quality             0
repeatability       0

Generated Visualizations:
1. Correlation Heatmap: media\heatmap.png
2. Distribution of language: media\distribution.png


Error generating story with LLM: 

You tried to access openai.ChatCompletion, but this is no longer supported in openai>=1.0.0 - see the README at https://github.com/openai/openai-python for the API.

You can run `openai migrate` to automatically upgrade your codebase to use the 1.0.0 interface. 

Alternatively, you can pin your installation to the old version, e.g. `pip install openai==0.28`

A detailed migration guide is available here: https://github.com/openai/openai-python/discussions/742
