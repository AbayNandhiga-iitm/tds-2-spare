Dataset Name: goodreads

Basic Description:
        books_count  original_publication_year  language_code  average_rating  ratings_count  work_ratings_count  work_text_reviews_count      ratings_1      ratings_2      ratings_3     ratings_4     ratings_5
count  10000.000000                9979.000000            0.0    10000.000000   1.000000e+04        1.000000e+04             10000.000000   10000.000000   10000.000000   10000.000000  1.000000e+04  1.000000e+04
mean      75.712700                1981.987674            NaN        4.002191   5.400124e+04        5.968732e+04              2919.955300    1345.040600    3110.885000   11475.893800  1.996570e+04  2.378981e+04
std      170.470728                 152.576665            NaN        0.254427   1.573700e+05        1.678038e+05              6124.378132    6635.626263    9717.123578   28546.449183  5.144736e+04  7.976889e+04
min        1.000000               -1750.000000            NaN        2.470000   2.716000e+03        5.510000e+03                 3.000000      11.000000      30.000000     323.000000  7.500000e+02  7.540000e+02
25%       23.000000                1990.000000            NaN        3.850000   1.356875e+04        1.543875e+04               694.000000     196.000000     656.000000    3112.000000  5.405750e+03  5.334000e+03
50%       40.000000                2004.000000            NaN        4.020000   2.115550e+04        2.383250e+04              1402.000000     391.000000    1163.000000    4894.000000  8.269500e+03  8.836000e+03
75%       67.000000                2011.000000            NaN        4.180000   4.105350e+04        4.591500e+04              2744.250000     885.000000    2353.250000    9287.000000  1.602350e+04  1.730450e+04
max     3455.000000                2017.000000            NaN        4.820000   4.780653e+06        4.942365e+06            155254.000000  456191.000000  436802.000000  793319.000000  1.481305e+06  3.011543e+06

Missing Values:
books_count                      0
original_publication_year       21
language_code                10000
average_rating                   0
ratings_count                    0
work_ratings_count               0
work_text_reviews_count          0
ratings_1                        0
ratings_2                        0
ratings_3                        0
ratings_4                        0
ratings_5                        0

Generated Visualizations:
1. Correlation Heatmap: goodreads\heatmap.png
2. Distribution of books_count: goodreads\distribution.png


Error generating story with LLM: 

You tried to access openai.ChatCompletion, but this is no longer supported in openai>=1.0.0 - see the README at https://github.com/openai/openai-python for the API.

You can run `openai migrate` to automatically upgrade your codebase to use the 1.0.0 interface. 

Alternatively, you can pin your installation to the old version, e.g. `pip install openai==0.28`

A detailed migration guide is available here: https://github.com/openai/openai-python/discussions/742
