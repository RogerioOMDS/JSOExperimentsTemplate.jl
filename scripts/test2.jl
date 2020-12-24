using NLPModels, LinearAlgebra, JSOSolverTemplate, CUTEst, ForwardDiff

# f(x) = (x[1]- 1)^2 + 4 * (x[2] - 2)^2
f(x) = log(exp(-x[1]) + exp(-x[2]) + exp(x[1] + x[2]))
# f(x) = (x[1] -1)^2 + 100 * (x[2] - x[1]^2)^2


nlp = ADNLPModel(x->f(x), [-1.2; 1.0])

output = JSOSolverTemplate.Newton_rc_bissec(nlp)

println(output)

println(grad(nlp, output.solution))

finalize(nlp)

println("CUTE ROSEN")
nlp = CUTEstModel("ROSENBR")

output = JSOSolverTemplate.Newton_rc_bissec(nlp)

println(output)

println((grad(nlp, output.solution)))


finalize(nlp)