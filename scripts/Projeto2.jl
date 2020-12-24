#Método de Newton com região de confiança e método da bissecção
using LinearAlgebra;
using ForwardDiff;
# using JuMP, Ipopt;
using NLPModels

#-------------------------------------------------------------------
#  Método da bissecção próprio para Newton com região de confiança
#----------------------------------------------------------------
function Bisseccao(h)  
    a=0
    λ1=0
    b=1
    ϵ=1e-4
    status= :resolvido
    while h(λ1)>ϵ
        if h(a)*h(b)<0
            while abs(b-a) > ϵ
                λ1=(b+a)/2
                if h(λ1)*h(a)<0
                    b=λ1
                else h(λ1)*h(a)<0
                    a=λ1
                end
            end
        end 
        if h(a)*h(b)>0
            if b!=1001
                b=b+50
            elseif a < 2000
                a=a+40
            else
                status= :bisseccao_falhou
                break
            end
        end
    end
    return λ1 , status
end
# 
#teste(x)=(x-60)*(x-103)
#Bisseccao(teste)
#---------------------------------------------------------------
# Newton com região de confiança e bissecção sem NLP
#-------------------------------------------------------------
function newton_rc_bisseccao(
    f,
    x;
    Δ = 5.0, 
    ϵ=1e-4,
    η₁ = 1e-2,
    max_tempo=5.0,
    max_iter=10000
    )
    ∇f(x) = ForwardDiff.gradient(f, x)
    H(x) = ForwardDiff.hessian(f, x)   
    η₂ = 0.75
    n=length(x)
    d=zeros(n)
    iter = 0    
    t0=time()
    Δt=time()-t0
    resolvido=norm(∇f(x))< ϵ
    cansado=Δt > max_tempo || iter > max_iter
    
    while !(resolvido || cansado)
        fx = f(x)
        gx = ∇f(x)
        Hx = H(x)
        if norm(d) < Δ    
            d= Hx \ -gx 
        else
            g(λ)=norm(inv(Hx+λ*diagm(ones(n)))*gx)-Δ
            λ = Bisseccao(g)   
            d=(Hx+λ*diagm(ones(n)))\-gx  
        end
        Ared = fx - f(x + d)
        Pred = fx - (fx + dot(d, gx) + dot(d, Hx * d) / 2)
        ρ = Ared / Pred
        if ρ < η₁
            Δ = Δ / 2
        elseif ρ < η₂
            x = x + d
        else
            x = x + d
            Δ = 2Δ
        end
        iter += 1
        Δt=time()-t0
        resolvido=norm(∇f(x))< ϵ
        cansado=Δt > max_tempo || iter > max_iter
    end
    status = :desconhecido
    if resolvido
        status = :arrasou
    elseif cansado
        if Δt > max_tempo
            status = :max_tempo
        elseif iter > max_iter
            status= :max_iter
        end
    end
    
    return x, status, Δt, iter
end




#-----------------------------------------------------------
# Newton com região de confiança sem bissecção do prof Abel
#--------------------------------------------------------
# function newton_rc(f, x)
#     ∇f(x) = ForwardDiff.gradient(f, x)
#     H(x) = ForwardDiff.hessian(f, x)
#     η₁ = 1e-2
#     η₂ = 0.75
#     Δ = 1.0
#     iter = 0
#     while norm(∇f(x)) > 1e-6
#         fx = f(x)
#         gx = ∇f(x)
#         Hx = H(x)
#         model = Model(with_optimizer(Ipopt.Optimizer, print_level=0)) 
#         @variable(model, d[1:2])
#         @objective(model, Min, fx + dot(d, gx) + dot(d, Hx * d) / 2)
#         @NLconstraint(model, d[1]^2 + d[2]^2 ≤ Δ^2)
#         optimize!(model)
#         d = value.(d)

#         Ared = f(x) - f(x + d)
#         Pred = f(x) - (fx + dot(d, gx) + dot(d, Hx * d) / 2)
#         ρ = Ared / Pred
#         if ρ < η₁
#             Δ = Δ / 2
#         elseif ρ < η₂
#             x = x + d
#         else
#             x = x + d
#             Δ = 2Δ
#         end

#         iter += 1
#         if iter > 1000
#             error("Nao converge")
#         end
#     end

#     return x, iter
# end


#-----------------------------------------------------------------
#                                 TESTES!
#----------------------------------------------------------------
teste1(x) = log(exp(-x[1]) + exp(-x[2]) + exp(x[1] + x[2]))
newton_rc_bisseccao(teste1, [-1;-1])

teste2(x)=(x[1]-1)^2+4*(x[2]-x[1]^2)^2
newton_rc_bisseccao(teste2, [-1.2;1.0])




