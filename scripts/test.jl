using NLPModels, LinearAlgebra, JSOSolverTemplate, CUTEst, ForwardDiff

# problemas 2 var, 0 restriçẽos, variáveis livres
# "DIXMAANI"
# "LUKVLI7"
# "GAUSS2"
# "LIARWHD"
# "SCHMVETT"
# "LUKSAN13LS"
# "POLAK4"
# "EXPFITA"
# "VAREIGVL"
# "MSS1"
# "WAYSEA1NE"
# "FBRAIN2"
# "BROWNDENE"

# f(x) = x[1]^2 

f(x) = (x[1]- 1)^2 + 4 * (x[2] - 2)^2
# f(x) = log(exp(-x[1]) + exp(-x[2]) + exp(x[1] + x[2]))
# f(x) = (x[1] -1)^2 + 100 * (x[2] - x[1]^2)^2
# ∇f(x) = ForwardDiff.gradient(f, x)
# x, k = JSOSolverTemplate.L_BFGS(f, [-1.2; 1.0], m = 3)
# println("x = $x e k = $k")

nlp = ADNLPModel(x->f(x), ones(2))

output = JSOSolverTemplate.LBFGS_StrongWolfe(nlp)

println("Minha Rsenbrock")
println(output)
println(grad(nlp, output.solution))

finalize(nlp)
println("##############################################################################################################################################")

println("CUTE ROSEN")
nlp = CUTEstModel("AKIVA")

finalize(nlp)
g = grad(nlp, [1.0,1.0])
pirntln(g)

# output = JSOSolverTemplate.LBFGS_StrongWolfe(nlp)

# println(output)

# println((grad(nlp, output.solution)))
