using LinearAlgebra

# (centered) finite differences gradient checker. (check whether calculated jacobian is correct)
# Inputs x∈ℝ^n   function f: ℝ^n->ℝ^d    function df: ℝ^n -> ℝ^(d×n) 
x = [1, 2, 3]
f = (x,y,z) -> [x^2 + y, x - y^2+z]
df = (x, y, z) -> [2x 1 0;
                   1 -2y 1]

#determine n and d and store basis vectors
n = 3   # length(Base.invokelatest(Base.methodswith(f))[1].sig.parameters)
d = 2   # length(f(0, 0, 0))
basisvectors = Matrix(I, n, n)

# init J and eps
J_hat = zeros(d,n)
eps = 10e-6

# calculate J with centered finite differences
for i in 1:n
    J_hat[:, i] = (f((x + eps * basisvectors[:, i])...) - f((x - eps * basisvectors[:, i])...)) / (2 * eps)
end


if norm(J_hat - df(x...), Inf) < 1e-4
    println("True")
else
    println("False")
end