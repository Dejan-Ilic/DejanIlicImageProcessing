
#naive_boxfilter(m::AbstractMatrix, r::Int) = imfilter(m, centered(ones(2r+1,2r+1)/(2r+1)^2))
function naive_boxfilter(img, r)
	res = similar(img)
	out = similar(img)
	H, W = size(img)

	#horizontal blur
	@inbounds Threads.@threads for j=1:W
		res[1,j] = @views sum(img[1:1+r, j]) + 0*r*img[1,j]
		for i=2:H
			add = i+r > H ? 0*img[H, j] : img[i+r, j]
			sub = i-1-r < 1 ? 0*img[1,j] : img[i-1-r, j]
			res[i,j] = res[i-1,j] - sub + add
		end
	end

	#vertical blur
	@inbounds Threads.@threads for i=1:H
		out[i,1] = @views sum(res[i, 1:1+r]) + 0*r*res[i,1]
		for j=2:W #note: removing the :? and replacing it with 3 loops yields no significant gains
			add = j+r > W ? 0*res[i, W] : res[i, j+r]
			sub = j-1-r < 1 ? 0*res[i, 1] : res[i, j-1-r]
			out[i,j] = out[i,j-1] - sub + add
		end
	end

	w = 2r+1
	#out .= out ./ w^2
	return out
end


#implements guided filtering based on [MATLAB code I found](https://github.com/clarkzjw/GuidedFilter)
function naive_guided_filter(I, p, r=2, eps=1e-2)
	N = naive_boxfilter(ones(eltype(I), size(I)), r)

	mean_I = naive_boxfilter(I, r) ./ N
	mean_p = naive_boxfilter(p, r)  ./ N
	mean_Ip = naive_boxfilter(I.*p, r) ./N
	cov_Ip = mean_Ip - mean_I .* mean_p;

	mean_II = naive_boxfilter(I.*I, r) ./ N
	var_I = mean_II - mean_I .* mean_I;

	a = cov_Ip ./ (var_I .+ eps);
	b = mean_p - a .* mean_I;

	mean_a = naive_boxfilter(a, r) ./ N
	mean_b = naive_boxfilter(b, r) ./ N

	q = mean_a .* I + mean_b

	return q
end

function naive_guided_fusion(stack::AbstractArray{T, 3}, r1=45, ε1=0.3, r2=7, ε2=1e-6) where T
	H,W,L = size(stack)

	absH = similar(stack)
	for l=1:L
		@views imfilter!(absH[:,:,l], stack[:,:,l], Kernel.Laplacian())
	end
	absH .= abs.(absH)

	S = similar(stack)
	for l=1:L
		@views imfilter!(S[:,:,l], absH[:,:,l],  Kernel.gaussian((5,5),(11,11))) #5x5 kernel
	end
	S .+= 1e-12
	S .= S ./ sum(S, dims=3)


	P = zeros(T,size(stack))
	for j=1:W, i=1:H
		m_ij = @views maximum(S[i,j,:])
		for l=1:L
			if S[i,j,l]==m_ij
				P[i,j,l] = one(T)
				break #only the first occurance like in matlab
			end
		end
	end

	W_B = similar(stack)
	W_D = similar(stack)

	#guided filtering
	for l=1:L
		W_B[:,:,l] .= @views naive_guided_filter(stack[:,:,l], P[:,:,l], r1, ε1)
		W_D[:,:,l] .= @views naive_guided_filter(stack[:,:,l], P[:,:,l], r2, ε2)
	end

	#they do this in their implementation
	@. W_B = min(one(T), max(zero(T), W_B)) + 1e-12
	@. W_D = min(one(T), max(zero(T), W_D)) + 1e-12

	#normalize
	for j=1:W, i=1:H
		s_ij = sum(@views W_B[i,j,:])
		W_B[i,j, :] ./= s_ij
	end
	for j=1:W, i=1:H
		s_ij = sum(@views W_D[i,j,:])
		W_D[i,j, :] ./= s_ij
	end

	#fusion
	B = similar(stack)
	ftr = centered(ones(T,31,31))/(31*31)
	for l=1:L
		@views imfilter!(B[:,:,l], stack[:,:,l], ftr)
	end
	D = stack - B

	B_bar = sum(W_B .* B, dims=3)
	D_bar = sum(W_D .* D, dims=3)

	return min.(one(eltype(B_bar)), max.(zero(eltype(B_bar)), dropdims(B_bar + D_bar, dims=3)))
end

function boxfilter!(out, res, img, r)
	H, W = size(img)

	#horizontal blur
	@inbounds Threads.@threads for j=1:W
		res[1,j] = @views sum(img[1:1+r, j]) + r*img[1,j]
		for i=2:H
			add = i+r > H ? img[H, j] : img[i+r, j]
			sub = i-1-r < 1 ? img[1,j] : img[i-1-r, j]
			res[i,j] = res[i-1,j] - sub + add
		end
	end

	#vertical blur
	@inbounds Threads.@threads for i=1:H
		out[i,1] = @views sum(res[i, 1:1+r]) + r*res[i,1]
		for j=2:W #note: removing the :? and replacing it with 3 loops yields no significant gains
			add = j+r > W ? res[i, W] : res[i, j+r]
			sub = j-1-r < 1 ? res[i, 1] : res[i, j-1-r]
			out[i,j] = out[i,j-1] - sub + add
		end
	end

	w = 2r+1
	out .= out ./ w^2
	return out
end

function guided_filter!(out, l1, l2, l3, l4, l5, I, p, boxw::Int=2, eps=1e-2)
	mean_I = l1#alloc l1
	mean_p = l2#alloc l2
	mean_Ip = l3#alloc l3
	Ip = l4#alloc l4
	Ip .= I.*p

	boxfilter!(mean_I, out, I, boxw)
	boxfilter!(mean_p, out,  p, boxw)
	boxfilter!(mean_Ip, out, Ip, boxw)

	cov_Ip = mean_Ip#alloc l3
	cov_Ip .= mean_Ip .- mean_I .* mean_p;

	II = Ip#alloc l4
	II .= I.*I

	mean_II = out#alloc out
	boxfilter!(mean_II, l5, II, boxw)#overwrite l5

	var_I = mean_II#alloc out
	var_I .= mean_II .- mean_I .* mean_I;

	a = var_I#alloc out
	b = mean_p#alloc l2

	a .= cov_Ip ./ (var_I .+ eps);
	b .= mean_p .- a .* mean_I;

	mean_a = l1#alloc l1
	mean_b = l3#alloc l3
	boxfilter!(mean_a, l4, a, boxw)#overwrite l4
	boxfilter!(mean_b, l4, b, boxw)#overwrite l4

	out .= mean_a .* I .+ mean_b

	return out
end

function guided_filter(I,p,boxw::Int=2,eps=1e-2)
	out = similar(p)
	l1=similar(p)
	l2=similar(p)
	l3=similar(p)
	l4=similar(p)
	l5=similar(p)
	guided_filter!(out,l1,l2,l3,l4,l5,I,p,boxw,eps)
	return out
end

"""
	guided_fusion(stack, r1=45, ϵ1=0.3, r2=7, ε2=1e-6)

Performs guided_fusion on `stack::Array{3}` and outputs `im::Array{2}`.
Internally, it splits into approximation and detail coefficients.
The parameters `r1` and `ϵ1` are for approximation coefficients.
"""
function guided_fusion(stack::AbstractArray{T, 3}, r1=45, ε1=0.3, r2=7, ε2=1e-6) where {T}
    	H,W,L = size(stack)

    	l1 = Matrix{T}(undef, H,W)
    	l2 = similar(l1)
    	l3 = similar(l1)
    	l4 = similar(l1)
    	l5 = similar(l1)

    	absH = similar(stack)
    	@inbounds for l=1:L
    		@views imfilter!(absH[:,:,l], stack[:,:,l], Kernel.Laplacian())
    	end
    	absH .= abs.(absH)

    	S = similar(stack)
		gauk = Kernel.gaussian((5,5),(11,11))
    	@inbounds for l=1:L #most memory allocations, but barely performance cost
    		@views imfilter!(S[:,:,l], absH[:,:,l], gauk) #5x5 kernel
    	end

    	P = absH #recycle
    	@inbounds for j=1:W, i=1:H
    		m_ij = S[i,j,1]
    		for l=2:L
    			if S[i,j,l]>m_ij
    				m_ij = S[i,j,l]
    			end
    		end

    		for l=1:L
    			P[i,j,l] = S[i,j,l]==m_ij ? one(T) : zero(T) #they only do this for the lowest l but that is ridiculous
    		end
    	end

    	W_B = S #recycle  #similar(stack)
    	W_D = similar(stack)

    	#guided filtering  #slowest part of the code
    	@inbounds for l=1:L
    		@views guided_filter!(W_B[:,:,l], l1,l2,l3,l4,l5, stack[:,:,l], P[:,:,l], r1, ε1)
    		@views guided_filter!(W_D[:,:,l], l1,l2,l3,l4,l5, stack[:,:,l], P[:,:,l], r2, ε2)
    	end

		@. W_B =  max(zero(T), W_B) + 1e-12 #they also bound below 1 but that's pointless
		@. W_D =  max(zero(T), W_D) + 1e-12


    	#normalize
    	@inbounds for j=1:W, i=1:H
    		s_ij = W_B[i,j,1]
    		for l=2:L
    			s_ij += W_B[i,j,l]
    		end
    		if s_ij != 0
    			@views W_B[i,j, :] ./= s_ij
    		end
    	end
    	@inbounds for j=1:W, i=1:H
    		s_ij = W_D[i,j,1]
    		for l=2:L
    			s_ij += W_D[i,j,l]
    		end
    		if s_ij != 0
    			@views W_D[i,j, :] ./= s_ij
    		end
    	end

    	#fusion
		B = similar(stack)

		@inbounds for l=1:L
    		@views boxfilter!(B[:,:,l], l1, stack[:,:,l], 15) #note: their boxfilter uses different boundary conditions
    	end
    	D = stack - B

    	B .= W_B .* B
    	D .= W_D .* D

		B_bar = l1
    	D_bar = l2
    	out = l3

    	sum!(B_bar, B)
    	sum!(D_bar, D)

    	out .= min.(one(T), max.(zero(T), B_bar .+ D_bar))
    	return out
    end
