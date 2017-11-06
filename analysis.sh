python analysis.py
Rscript models.R
Rscript make_graphics.R
for ((i=1; i<=19; i++))
do
  python analysis.py $i
  Rscript models.R
done
Rscript fisher_combined.R
