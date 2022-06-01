function read_oocyte_stack!(stack::Array{Float64,3}, main_folder::String)
	i = 0;

	for F=-75:15:75
		i += 1
		path = joinpath(main_folder, "F$F")
		imgname = readdir(path)
		if length(imgname) < 1
			println("error on F$(F), no img")
			stack[:,:,i] .= -1 #signals error
		else
			stack[:,:,i] .= load(joinpath(path,imgname[1]))
		end
	end

	return stack
end

function read_oocyte_stack(main_folder::String, h=800, w=800, z=11)
	stack = Array{Float64, 3}(undef, h, w, z)

	return read_oocyte_stack!(stack, main_folder)
end

function read_oocyte_stack(n::Int, OOCYTE_FOLDER, OOCYTE_LIST)
	if n < 1 || n > length(OOCYTE_LIST)
		throw(DomainError(n, "Out of range: n should be between 1 and $(length(OOCYTE_LIST))"))
	else
        path = joinpath(OOCYTE_FOLDER,OOCYTE_LIST[n])
		read_oocyte_stack(path)
	end
end

function oocyte_file_unpack(main_folder)
	for F=-75:15:75
		path = joinpath(main_folder, "F$F")
		imgname = readdir(path)
		if length(imgname) < 1
			println("error on F$(F), no img")
		else
			cp(joinpath(path,imgname[1]), joinpath(main_folder, "im$F.JPG"))
		end
	end
end

#(https://juliaimages.org/ImageFeatures.jl/dev/function_reference/
#RADIUS DETERMINED EMPIRICALLY
function cell_circle_detection(img, p1=98, p2=80, MIN_CELL_RADIUS = 140, MAX_CELL_RADIUS = 210)
	img_edges = canny(img, (Percentile(p1), Percentile(p2)));
	dx, dy=imgradients(img, KernelFactors.ando5);
	img_phase = phase(dx, dy);

	centers, radii = hough_circle_gradient(img_edges, img_phase, MIN_CELL_RADIUS:MAX_CELL_RADIUS);

	return centers, radii
end

function detect_and_mark_circles(img, p1, p2, m, M)
	H,W = size(img)
	centers, radii = cell_circle_detection(img, p1, p2, m, M)
	colimg = RGB.(img)

	red = RGB(1,0,1)
	smartstep = 360/(2*π*M)/2
	for c in eachindex(centers)
		colimg[centers[c]] = red
		for θ = 0:smartstep:360
			i = round(Int, radii[c] * sin(θ) + centers[c][1] + 0.5)
			j = round(Int, radii[c] * cos(θ) + centers[c][2] + 0.5)

			i = max(1, min(H, i))
			j = max(1, min(W, j))

			colimg[i,j] = red
		end
	end

	colimg
end

function crop_cell(img, center, cropsize::Int=512)
	H, W = size(img)
	d = cropsize÷2

	ci = max(1 + d, min(H-d, center[1]))
	cj = max(1 + d, min(W-d, center[2]))

	return img[ci-d:ci+d-1, cj-d:cj+d-1]
end

function preprocess_cell(img::Matrix{T}, cropsize::Int=512) where {T}
	centers, radii = cell_circle_detection(img)
	if length(centers) == 0
		@info "No circles found, changing params..."
		#todo
		return nothing
	end

	nc = length(centers)
	center = (sum(c[1] for c in centers)÷nc,
			  sum(c[2] for c in centers)÷nc)

	return crop_cell(img, center, cropsize)
end

function preprocess_cell(stack::Array{T, 3}, cropsize::Int=512) where {T}
	H,W,L = size(stack)

	center = CartesianIndex(0,0)
	nc = 0
	for l=1:L
		centers, radii = cell_circle_detection(@view(stack[:,:,l]))
		center += sum(centers)
		nc += length(centers)
	end

	if nc == 0
		@info "No circles found in any layer"
		#todo
	end

	return CartesianIndex(center[1]÷nc, center[2]÷nc)
end

function flatten_oocyte_folder(src::String, dest::String)
	mkdir(dest)
	for F=-75:15:75
		folder = joinpath(src, "F$F")
		ims = readdir(folder)
		if length(ims) == 0
			@error "$folder is empty"
		else
			if length(ims) > 1
				@warn "Found multiple images in $folder"
			end
			im = joinpath(folder, ims[1])

			cp(im, joinpath(dest, "F$F.jpg"))
		end
	end
end

function flatten_oocyte_folder(n::Int, OOCYTE_FOLDER, OOCYTE_LIST, dest::String="")
	if n < 1 || n > length(OOCYTE_LIST)
		@error "index out of bounds, should be at most $(length(OOCYTE_LIST))"
	end

	flatten_oocyte_folder(joinpath(OOCYTE_FOLDER, OOCYTE_LIST[n]), joinpath(dest, OOCYTE_LIST[n]))
end
