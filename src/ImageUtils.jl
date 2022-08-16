
imshow(im::AbstractMatrix{T}) where {T} = reinterpret(Gray{T}, im)

sliceshow(im::Array{T, 3}, slice::Int) where {T} = imshow(@view(im[:,:,slice]))

graytofloat(im::AbstractMatrix{Gray{T}}) where {T} = reinterpret(T, im)

function imrescale(x::AbstractMatrix)
	m=minimum(x)
	M=maximum(x)

	@. Gray( (x-m)/(M-m))
end

abslogrescale(x) = abs.(x) .+ 1 .|> log |> imrescale
