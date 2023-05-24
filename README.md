# EPA-simmodel


To know:

-We moeten een problem formulation kiezen. Welke porblem formulation hangt af van wat we willen analyseren (voor wie: onself of de tegenpartij).

-Het systeem begrijpen, waar is het sensitief voor, wat zijn de meest fundamental variabelen. Robustness
-Welke soort sampling gaan we gebruiken: LHS of SoBol
-Welke sensitivity analysis: global/regional: LInear Regression, Sobol, feature scoring(extra tree) 
    - Dit zonder policies. En een keer met policies?
-Scenario Discovery: Prim (of CART) of Dimensional Stacking (Subspace partitioning parameters: threshold, peeling_factor)
    -Prim: welke subspace willen we: hoeveel coverage and density