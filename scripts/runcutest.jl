using CUTEst
using JSOSolvers
using JSOSolverTemplate
using LinearAlgebra
using NLPModels
using Plots
using SolverBenchmark
using CSV

n_list = [2,10,1000]

for n in n_list

function runcutest(n)
  pnames = CUTEst.select(max_var=n, max_con=0, only_free_var=true)
  sort!(pnames)
  problems = (CUTEstModel(p) for p in pnames) # Generator of problems

  # If you need to define different arguments, you can wrap
  # trunk_wrapper(nlp; kwargs...) = trunk(nlp, max_time=3.0; kwargs...)
  # yoursolver_wrapper(nlp; kwargs...) = LBFGS_StrongWolfe(nlp, m = 3; kwargs...)
  # yoursolver_wrapper2(nlp; kwargs...) = LBFGS_StrongWolfe(nlp, m = 17; kwargs...)
  # yoursolver_wrapper3(nlp; kwargs...) = nlp_L_BFGS(nlp, max_eval = 1000; kwargs...)
  # yoursolver_wrapper4(nlp; kwargs...) = nlp_newton_rc_bissec(nlp, max_bissec = 1000  ; kwargs...)
  # yoursolver_wrapper5(nlp; kwargs...) = nlp_newton_rc_bissec(nlp, Δ = 10.0; kwargs...)

  solvers = Dict(
    :lbfgs => lbfgs,
    # :trunk => trunk,
    :LBFGS_StrongWolfe_m3 => LBFGS_StrongWolfe,
    # :Newton_rc_bissec => Newton_rc_bissec,
    # :LBFGS_StrongWolfe_m3 => yoursolver_wrapper,
    # :Δ_10 => yoursolver_wrapper5
    # :eval_1000 => yoursolver_wrapper,
    # :LBFGS_StrongWolfe_m17 => yoursolver_wrapper2
    # :eval_1000 => yoursolver_wrapper3
    # :nlp_L_BFGS => nlp_L_BFGS
  )

  stats = bmark_solvers(solvers, problems)
end

function table_and_plots(stats)
  cols = [:name, :nvar, :status, :objective, :dual_feas, :elapsed_time, :neval_obj, :neval_grad, :neval_hess]
  for (k,v) in stats
    open("tabelas/tabela-$(k)_$(n).md", "w") do io
      pretty_stats(io, v[:,cols])
    end
    open("tabelas/tabela-$(k)_$(n).tex", "w") do io
      pretty_latex_stats(io, v[:,cols])
    end
  end

  # Performance profile

  # First, define metrics
  not_first_order(df) = df.status .!= :first_order
  Fx = hcat([df.objective for df in values(stats)]...)
  Fx[isnan.(Fx)] .= Inf
  fmin = minimum(Fx, dims=2)
  not_best_fx(df) = df.dual_feas .> fmin * 1.01 .+ 1e-6 # 1% tolerance, with safeguard for 0.
  costs = [
    df -> df.elapsed_time,
    df -> not_first_order(df) * Inf + df.elapsed_time,
    df -> not_best_fx(df) * Inf + df.elapsed_time,
    df -> not_first_order(df) * Inf + df.neval_obj + df.neval_grad + df.neval_hess + df.neval_hprod
  ]
  costnames = [
    "Pure Δt",
    "Δt checking 1st order",
    "Δt checking fx ≈ fmin",
    "Eval checking 1st order"
  ]
  p = profile_solvers(stats, costs, costnames)
  png(p, "plots/profile_$n")
end

# You can comment out this after the tests are run
stats = runcutest(n)
p = table_and_plots(stats)

# CSV.write("Rogerio_lbfgs_var_$n.csv", stats[:LBFGS_StrongWolfe])
# CSV.write("Rogerio_lbfgs_m_17_var_$n.csv", stats[:LBFGS_StrongWolfe_m17])
# CSV.write("Rafaela_Newton_var_$n.csv", stats[:Newton_rc_bissec])

end