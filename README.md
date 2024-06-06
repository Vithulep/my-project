# CROP : CROp Planning System

About the project:
A mismatch between the crops produced by farmers and the respective market demands could
potentially lead to large-scale crop dumping. This has been observed year after year in many
countries. This regularly leads to huge financial losses and distress to the farmers. To alleviate
this problem, we address the macro-level problem of district level or county level agricultural
crop planning in any given state. Our interest is in how the Government or any state administration
could make an informed recommendation on which crop acreages (number of acres
cultivated under each crop) to allocate in which districts or geospatial regions in a given state,
so as to match the demand for the crops and maximize the profits for the farmers. To this end,
we design and develop CROP-S (CROp Planning System) to determine an assignment of crop
acreages to districts so as to maximize the profits for the farmers while simultaneously ensuring
required crop security levels for each district. CROP-S uses data about predicted demands,
transportation costs, compliance ratios (fraction of farmers who will follow the recommended
crop plan), and historical data about yields and prices to arrive at an optimal allocation of
crop acreages to districts. Two different CROP-S models are developed in this work. The first
model uses a methodology based on genetic algorithms. The second model uses a mathematical
programming approach for maximizing the profits of farmers taking into account key determinants
of farmer profits. We believe CROP-S provides an effective, decision support tool for the
Government to issue crop recommendations to the district administrations, who in turn issue
advisories to farmers. We demonstrate the effectiveness of CROP-S using real-world data of
the top 12 crops grown in 30 districts of Karnataka state in India. We additionally introduce
the concept of shaded allocation, wherein a non-optimal allocation is recommended such that,
after compensating for the effects of the compliance ratio, the actual acreage is near optimal.

## Run
To run locally

    cd CROP
    do pip install -r requirements.txt     ->to install all the requirements 
    src_old/main.py examples/test1
    if this dosen't work do: python src/main.py examples/test1

To plot

    cd CROP
    src_old/plot.py examples/test1
    if this dosen't work do: python src/plot.py examples/test1

src_new -> Contains optimization with gurobi:
    
    cd CROP
    src_new/gurobi.py examples/test1
    
