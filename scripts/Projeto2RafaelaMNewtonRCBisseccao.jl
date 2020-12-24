using LinearAlgebra;
using ForwardDiff;
# using JuMP, Ipopt;
using NLPModels
using SolverTools

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


#---------------------------------------------------------------
# Newton com região de confiança e bissecção com NLP
#-------------------------------------------------------------
function newton_rc_bisseccao(
    nlp :: AbstractNLPModel;
    Δ = 5.0, 
    ϵ=1e-4,
    η₁ = 1e-2,
    max_tempo=10.0,
    max_iter=10000
    )

    x=copy(nlp.meta.x0)
    f(x)=obj(nlp,x)
    ∇f(x) = grad(nlp,x)
    H(x) = ForwardDiff.hessian(f, x) 
    fx=f(x)
    gx=∇f(x)  
    η₂ = 0.75
    n=length(x)
    d=zeros(n)
    iter = 0    
    t0=time()
    Δt=time()-t0

    status = :unknown
    resolvido=norm(gx)< ϵ
    cansado=Δt > max_tempo || iter > max_iter

    while !(resolvido || cansado)
        println("entrei")
        fx = f(x)
        gx = ∇f(x)
        Hx = H(x)
        if norm(d) < Δ    
            d= Hx \ -gx 
            println("agora entrei aqui")
        else
            g(λ)=norm(inv(Hx+λ*diagm(ones(n)))*gx)-Δ
            if Bisseccao(g)[2] != :bisseccao_falhou
                
                λ = Bisseccao(g)[1]   
                println("λ = $λ")
                d=(Hx+λ*diagm(ones(n)))\-gx  
            else
                status = :exception
                break
            end
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
    
    if resolvido
        status = :first_order
    elseif cansado
        if Δt > max_tempo
            status = :max_time
        elseif iter > max_iter
            status= :max_iter
        end
        
    end
    
    
    return GenericExecutionStats(status, nlp, objective=fx, solution=x,
    dual_feas=norm(gx),iter=iter,elapsed_time=Δt)
    
end


# teste(x) = (x[1]- 1)^2 + 4 * (x[2] - 2)^2 #  Defino uma função
teste(x) = log(exp(-x[1]) + exp(-x[2]) + exp(x[1] + x[2]))
#teste(x) = (x[1] -1)^2 + 100 * (x[2] - x[1]^2)^2

nlp = ADNLPModel(x->teste(x), ones(2)) # Passo ela pro  nlp + ponto inicial

output = newton_rc_bisseccao(nlp,max_iter=100) # Aqui o seu algoritmo 

println(output) # confere os status de resolução

println(grad(nlp, output.solution)) # chamo o gradiente com a solução que encontrou pra saber se encontrou um valor pequeno

finalize(nlp) # finalizo o nlp
