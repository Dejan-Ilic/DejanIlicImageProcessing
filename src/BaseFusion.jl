
function base_fusion(stack::Array{T,3},hws::Int=10,gamma=100) where T
	ws = 2*hws+1;
	kernel = centered(ones(eltype(stack), (ws, ws)));
	padding = Pad(:replicate)

	#pre allocate weights
	H,W,L = size(stack);

	weights = similar(stack)
	grad = similar(stack, (H,W))
	tv = similar(grad)

	calc_grad(c, c_di, c_dj) = sqrt((c - c_di)^2 + (c - c_dj)^2)
	o = one(eltype(stack))

	@inbounds for l = 1:L
		Threads.@threads for j=1:W-1
			@simd for i=1:H-1
				grad[i,j] = calc_grad(stack[i,j,l], stack[i+1,j,l], stack[i,j+1,l])
			end

			grad[H,j] = calc_grad(stack[H,j,l], o, stack[H,j+1,l])
		end

		#j = W
			@simd for i=1:H-1
				grad[i,W] = calc_grad(stack[i,W,l], stack[i+1,W,l], o)
			end
		#end
		grad[H,W] = calc_grad(stack[H,W,l], o, o)

		#pixels should have in/out of focus properties similar to their surroundings so blur the grad image
		imfilter!(tv, grad,kernel, padding)
		weights[:,:,l] .= tv;
	end

	#enhance weights
	we_max, map = findmax(weights,dims=3);
	we_max .= max.(we_max, 1e-6); #avoid zero we_max

	#dejan: boost the maximum weight with a factor gamma
	weights[map] .= we_max .* gamma

	x = tv
	return weighted_stack_combine_sc!!!(x, weights, we_max, stack)
end


function naive_base_fusion(stack,hws::Int=10,gamma=100)
	ws = 2*hws+1;
	kernel = centered(ones(eltype(stack), (ws, ws)));
	f1 = centered( [0 0 0; 0 1.0 -1.0; 0 0 0]);
	f2 = centered( [0 0 0; 0 1.0 0;    0 -1.0 0]);

	H,W,L = size(stack);

	#pre allocate weights
	weights = similar(stack)

	dx = similar(stack, (H,W))
	dy = similar(dx)
	grad = similar(dx)
	tv = similar(dx)

	@inbounds for l = 1:L
		imfilter!(dx, @view(stack[:,:,l]),f1,Fill(1,f1))
		imfilter!(dy, @view(stack[:,:,l]),f2,Fill(1,f2))

		for idx in eachindex(grad)
			grad[idx] = sqrt(dx[idx]^2+dy[idx]^2);
		end

		#pixels should have in/out of focus properties similar to their surroundings so blur the grad image
		imfilter!(tv, grad,kernel,Pad(:replicate))
		weights[:,:,l] .= tv;
	end

	#enhance weights
	we_max, map = findmax(weights,dims=3);
	we_max .= max.(we_max, 1e-6); #avoid zero we_max

	#dejan: boost the maximum weight with a factor gamma
	@inbounds for j = 1:W, i = 1:H
		weights[map[i,j]] = we_max[i,j]*gamma;
		#unless weights[i,j,:] is all 0, we_max[i,j] == weights[map[i,j]]
	end

	return weighted_stack_combine_sc!(weights,stack)
end

function weighted_stack_combine_sc!!!(x, weights, we_sum, stack)
	L = size(stack, 3)

	sum!(we_sum, weights)
	x .= zero(eltype(x))

	@inbounds for layer = 1:L
		#normalize the weights so ∑ l: 1→N (weights[i,j, l]) = 1 for all i,j
		weights[:,:,layer] .= @views weights[:,:,layer]./we_sum;
		x[:,:] .= @views x[:,:] .+ weights[:,:,layer].*stack[:,:,layer];
	end
	return x
end

function weighted_stack_combine_sc!(weights, stack)
	H,W,L = size(stack)
	x = zeros(eltype(weights), (H,W))
	we_sum = Array{eltype(weights), 2}(undef, H,W) #sum of weights along stack-axis

	return weighted_stack_combine_sc!!!(x, weights, we_sum, stack)
end
