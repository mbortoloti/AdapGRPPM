#
# Script for testing PPMNN and adapt-PPMNN algorithms
#
echo "Initializing tests..."
#
#echo "Testing Example 01"
#julia EXEMPLO_01.jl 2 2 1 1 "n_10_X0lognIn_"
#echo "End of Example 01"
#echo "Testing Example 01 B"
#julia EXEMPLO_01B.jl 50 50  1 1 "CostAnalysis"
#echo "End of Example 01"
#
#echo "Testing Example 02"
#julia EXEMPLO_02.jl 5 20 1 10 "n_5_20_X0_Rd_"
#echo "End of Example 02"
#
echo "Testing Example 03" 
julia example623.jl 10 100 5 1 "AllDim_X0In_"
echo "End of Example 03"
#
#echo "Testing Example 04"
#julia EXEMPLO_04.jl 2 2 1 1 "AllDim_X0Rd_"
#echo "End of Example 04"
