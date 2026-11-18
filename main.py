import experiments
import plots
import validation
print("1. Experimentos")
experiments.main()
print("2. Figuras")
plots.generar_todo()
print("3. Validación contra scikit-learn")
validation.validar()